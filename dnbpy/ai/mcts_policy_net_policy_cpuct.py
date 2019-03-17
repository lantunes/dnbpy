from .mcts_game import MCTSGame
from dnbpy import *
from math import *
import math


class MCTSPolicyNetPolicyCpuct:
    def __init__(self, board_size, num_playouts, cpuct, default_policy=None, normalize_policy_probs_with_softmax=True):
        self._board_size = board_size
        self._num_playouts = num_playouts
        self._cpuct = cpuct
        self._default_policy = default_policy
        self._normalize_policy_probs_with_softmax = normalize_policy_probs_with_softmax

    def set_num_playouts(self, num_playouts):
        self._num_playouts = num_playouts

    def select_edge(self, board_state, root_player_score, policy_net):
        root_node = self._search(board_state, root_player_score, policy_net)

        # return the move that was most visited
        most_visited_node = sorted(root_node.children, key = lambda c: c.visits)[-1]
        return self._get_selected_index(most_visited_node.state, board_state)

    def get_action_probs(self, board_state, root_player_score, policy_net, temperature=1.):
        root_node = self._search(board_state, root_player_score, policy_net)

        action_visit_count_map = {}
        for child in root_node.children:
            action_visit_count_map[as_string(child.state)] = child.visits
        # action probs are proportional to exponentiated visit count: N(s,a)^1/T / Sum_b N(s,b)^1/T
        exponentiated_sum = np.sum(np.array(list(action_visit_count_map.values()))**(1./temperature))
        action_prob_map = {}
        for state in action_visit_count_map:
            action_prob_map[state] = (action_visit_count_map[state]**(1./temperature)) / exponentiated_sum
        return action_prob_map

    def _search(self, board_state, root_player_score, policy_net):
        root_node = _Node(board_state, self._cpuct)

        # Perform playouts
        for i in range(self._num_playouts):
            game = MCTSGame(self._board_size, players=['root', 'opponent'], board_state=board_state,
                            root_player_score=root_player_score)
            current_player = game.get_current_player()
            node = root_node

            root_player_states = []
            opponent_states = []

            # Select
            while not node.has_untried_moves() and node.has_children():
                prev_state = node.state
                node = node.select_child()
                if current_player == 'root':
                    root_player_states.append(node.state)
                else:
                    opponent_states.append(node.state)
                selected_index = self._get_selected_index(node.state, prev_state)
                current_player = game.select_edge(selected_index, current_player)

            # Expand
            if node.has_untried_moves():
                # TODO: should try using a temperature, Beta, when getting action probs
                action_prob_map = policy_net.get_action_probs(node.state, normalize_with_softmax=self._normalize_policy_probs_with_softmax)
                move_state = node.select_untried_move()
                if current_player == 'root':
                    root_player_states.append(move_state)
                else:
                    opponent_states.append(move_state)
                selected_index = self._get_selected_index(move_state, node.state)
                current_player = game.select_edge(selected_index, current_player)
                node = node.add_child(move_state, self._cpuct)
                node.prob = action_prob_map[as_string(node.state)]

            # Rollout
            current_board_state = game.get_board_state()
            while not game.is_finished():
                if self._default_policy is None:
                    next_index = self._select_next_move_randomly(current_board_state)
                else:
                    next_index = self._default_policy.select_edge(current_board_state)
                current_player = game.select_edge(next_index, current_player)
                current_board_state = game.get_board_state()

            # Backpropagate
            #   backpropagate from the expanded node and work back to the root node
            while node is not None:
                root_score = game.get_score('root')
                opponent_score = game.get_score('opponent')
                node.visits += 1
                if node.state in root_player_states:
                    node.wins += (1 if root_score > opponent_score else 0)
                if node.state in opponent_states:
                    node.wins += (1 if opponent_score > root_score else 0)
                node = node.parent

        return root_node

    def _get_selected_index(self, child_state, parent_state):
        diff = [x1 - x2 for (x1, x2) in zip(child_state, parent_state)]
        argmax = max(enumerate(diff), key=lambda x: x[1])[0]
        return argmax

    def _select_next_move_randomly(self, current_state):
        zero_indices = []
        for i in range(len(current_state)):
            if current_state[i] == 0:
                zero_indices.append(i)
        return random.choice(zero_indices)


class _Node:
    def __init__(self, state, cpuct, parent=None):
        self.state = state
        self._cpuct = cpuct
        self.wins = 0
        self.visits = 0
        self.prob = None
        self.parent = parent
        self.children = []
        self.untried_moves = self._get_child_states()

    def _get_zero_indices(self):
        zero_indices = []
        for i in range(len(self.state)):
            if self.state[i] == 0:
                zero_indices.append(i)
        return zero_indices

    def _get_new_state(self, existing_state, selected_index):
        new_state = [x for x in existing_state]
        new_state[selected_index] = 1
        return new_state

    def _get_child_states(self):
        child_states = []
        zero_indices = self._get_zero_indices()
        for zero_index in zero_indices:
            child_states.append(self._get_new_state(self.state, zero_index))
        return child_states

    def has_untried_moves(self):
        return self.untried_moves != []

    def select_untried_move(self):
        return random.choice(self.untried_moves)

    def add_child(self, child_state, cpuct):
        child = _Node(child_state, cpuct, parent=self)
        self.children.append(child)
        self.untried_moves.remove(child_state)
        return child

    def has_children(self):
        return self.children != []

    def select_child(self):
        highest_ucb1 = None
        selected_child_node = None
        for child_node in self.children:
            ucb1 = child_node.ucb1()
            if highest_ucb1 is None or highest_ucb1 < ucb1:
                highest_ucb1 = ucb1
                selected_child_node = child_node
        return selected_child_node

    def ucb1(self):
        if self.visits == 0:
            return math.inf
        if self.prob is None:
            raise Exception("node has no action prob: %s" % self.state)
        return self.wins/self.visits + self._cpuct*self.prob*(sqrt(self.parent.visits)/(1 + self.visits))

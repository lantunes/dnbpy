from .mcts_game import MCTSGame
from dnbpy import *
from math import *
import math


class MCTSPolicy:
    """
    The game tree is represented as an acyclic directed graph, where nodes, representing game states, are not
    repeated, and thus may have multiple parents.
    TODO add option to reset node_map when select_edge is called
    """
    def __init__(self, board_size, num_playouts):
        # a dictionary of binary state strings to the nodes that represent them
        self._node_map = {}
        self._board_size = board_size
        self._num_playouts = num_playouts

    def select_edge(self, board_state, root_player_score):
        # Perform playouts
        for i in range(self._num_playouts):
            game = MCTSGame(self._board_size, players=['root', 'opponent'], board_state=board_state,
                            root_player_score=root_player_score)
            current_player = game.get_current_player()
            root_nodes = []
            opponent_nodes = []

            start_node = self._get_or_create_node(board_state)

            parent_node = start_node
            while not self._is_terminal(parent_node.state):
                child_states = self._get_child_states(parent_node.state)
                child_nodes = self._get_child_nodes(child_states)

                highest_ucb1 = None
                selected_child_node = None
                for child_node in child_nodes:
                    ucb1 = child_node.ucb1(parent_node)
                    if highest_ucb1 is None or highest_ucb1 < ucb1:
                        highest_ucb1 = ucb1
                        selected_child_node = child_node

                if current_player == 'root':
                    root_nodes.append(selected_child_node)
                else:
                    opponent_nodes.append(selected_child_node)
                selected_index = self._get_selected_index(selected_child_node.state, parent_node.state)
                current_player = game.select_edge(selected_index, current_player)

                if selected_child_node.visits == 0:
                    # Rollout
                    current_board_state = selected_child_node.state
                    while not game.is_finished():
                        next_index = self._select_next_move_randomly(current_board_state)
                        current_player = game.select_edge(next_index, current_player)
                        current_board_state = game.get_board_state()
                    break
                else:
                    parent_node = selected_child_node

            if not game.is_finished():
                raise Exception("the game should be finished")

            # Backpropagation
            start_node.visits += 1
            root_score = game.get_score('root')
            opponent_score = game.get_score('opponent')
            for root_node in root_nodes:
                root_node.visits += 1
                root_node.wins += (1 if root_score > opponent_score else 0)
            for opponent_node in opponent_nodes:
                opponent_node.visits += 1
                opponent_node.wins += (1 if opponent_score > root_score else 0)

        # Return the index with highest win average
        child_states = self._get_child_states(board_state)
        child_nodes = self._get_child_nodes(child_states)
        highest_win_avg = None
        best_state = None
        for child_node in child_nodes:
            child_win_average = child_node.win_average()
            if highest_win_avg is None or highest_win_avg < child_win_average:
                highest_win_avg = child_win_average
                best_state = child_node.state
        return self._get_selected_index(best_state, board_state)

    def _get_or_create_node(self, state):
        state_key = as_string(state)
        if state_key not in self._node_map:
            node = _Node(state)
            self._node_map[state_key] = node
        return self._node_map[state_key]

    def _get_child_nodes(self, child_states):
        child_nodes = []
        for child_state in child_states:
            child_nodes.append(self._get_or_create_node(child_state))
        return child_nodes

    def _get_child_states(self, board_state):
        child_states = []
        zero_indices = self._get_zero_indices(board_state)
        for zero_index in zero_indices:
            child_states.append(self._get_new_state(board_state, zero_index))
        return child_states

    def _get_zero_indices(self, board_state):
        zero_indices = []
        for i in range(len(board_state)):
            if board_state[i] == 0:
                zero_indices.append(i)
        return zero_indices

    def _get_new_state(self, existing_state, selected_index):
        new_state = [x for x in existing_state]
        new_state[selected_index] = 1
        return new_state

    def _get_selected_index(self, child_state, parent_state):
        diff = [x1 - x2 for (x1, x2) in zip(child_state, parent_state)]
        argmax = max(enumerate(diff), key=lambda x: x[1])[0]
        return argmax

    def _is_initial(self, board_state):
        """
        :param board_state: the binary string representation of the board 
        :return: True if this is the initial game state, where no edges are selected
        """
        return sum(board_state) == 0

    def _is_terminal(self, board_state):
        return sum(board_state) == len(board_state)

    def _select_next_move_randomly(self, current_state):
        zero_indices = self._get_zero_indices(current_state)
        return random.choice(zero_indices)


class _Node:
    def __init__(self, state):
        self.state = state
        self.wins = 0
        self.visits = 0

    def ucb1(self, parent):
        if self.visits == 0:
            return math.inf
        return self.wins/self.visits + sqrt(2*log(parent.visits)/self.visits)

    def win_average(self):
        if self.visits == 0:
            return 0
        return self.wins/self.visits

from treelib import Tree
from dnbpy import *
import random
import math


class CausalEntropicPolicy(Policy):
    def __init__(self, board_size, max_sample_paths):
        self._board_size = board_size
        self._max_sample_paths = max_sample_paths

    def select_edge(self, board_state):
        zero_indices = self.get_zero_indices(board_state)
        best_value = None
        best_state_index = zero_indices[0]
        for zero_index in zero_indices:
            new_state, made_box = self.create_new_state_from(board_state, zero_index)
            if math.factorial(len(zero_indices)) < 10:
                tree = self.all_paths(new_state, my_move=made_box is True)
            else:
                tree = self.sample_paths(new_state, num_sample_paths=self._max_sample_paths, my_move=made_box is True)
            new_state_value = self.causal_path_entropy(tree)
            if best_value is None or (new_state_value is not None and new_state_value > best_value):
                best_value = new_state_value
                best_state_index = zero_index
        return best_state_index

    def get_zero_indices(self, board_state):
        zero_indices = []
        for i in range(len(board_state)):
            if board_state[i] == 0:
                zero_indices.append(i)
        return zero_indices

    def create_new_state_from(self, existing_state, edge_index):
        new_state = [x for x in existing_state]
        new_state[edge_index] = 1
        game = Game(self._board_size, players=[0, 1])
        game._board_state = [x for x in existing_state]
        _, boxes_made = game.select_edge(edge_index, 0)
        return new_state, (boxes_made > 0) # TODO compute if box is made

    def shannon_entropy(self, state_strings):
        """
        Calculates the Shannon entropy for the given state strings.
        :param state_strings: a list of symbols, such as ['0011', '0101', '0011', '1100']
        :return: a real number representing the Shannon entropy
        """
        symbols = dict.fromkeys(state_strings)
        symbol_probabilities = [float(state_strings.count(symbol)) / len(state_strings) for symbol in symbols]
        H = -sum([p_symbol * math.log(p_symbol, 2.0) for p_symbol in symbol_probabilities])
        return H + 0  # add 0 as a workaround so we don't end up with -0.0

    def sample_paths(self, root_state, num_sample_paths, tree=None, root_state_id=None, my_move=True):
        root_state_string = as_string(root_state)
        if root_state_id is None:
            root_state_id = root_state_string
        if tree is None:
            tree = Tree()
            tree.create_node(root_state_string, root_state_id)
        zero_indices = self.get_zero_indices(root_state)
        if len(zero_indices) > 0:
            for i in range(num_sample_paths):
                child_state, made_box = self.create_new_state_from(root_state, random.choice(zero_indices))
                child_state_string = as_string(child_state)
                child_state_id = root_state_id + '-' + child_state_string
                if not tree.contains(child_state_id):
                    tree.create_node(child_state_string, child_state_id, parent=root_state_id, data=my_move)
                self.sample_paths(child_state, num_sample_paths=1, tree=tree, root_state_id=child_state_id,
                                  my_move=my_move if made_box else not my_move)
        return tree

    def all_paths(self, root_state, tree=None, root_state_id=None, my_move=True):
        root_state_string = as_string(root_state)
        if root_state_id is None:
            root_state_id = root_state_string
        if tree is None:
            tree = Tree()
            tree.create_node(root_state_string, root_state_id)
        zero_indices = self.get_zero_indices(root_state)
        for i in zero_indices:
            child_state, made_box = self.create_new_state_from(root_state, i)
            child_state_string = as_string(child_state)
            child_state_id = root_state_id + '-' + child_state_string
            if not tree.contains(child_state_id):
                tree.create_node(child_state_string, child_state_id, parent=root_state_id, data=my_move)
            self.all_paths(child_state, tree=tree, root_state_id=child_state_id, my_move=my_move if made_box else not my_move)
        return tree

    def causal_path_entropy(self, tree):
        nodes_by_level = dict.fromkeys(range(1, tree.depth() + 1))
        all_nodes = tree.all_nodes()
        for node in all_nodes:
            level = tree.level(node.identifier)
            if level in nodes_by_level:
                if nodes_by_level[level] is None:
                    nodes_by_level[level] = []
                nodes_by_level[level].append(node)
        path_entropy = None
        for level in nodes_by_level:
            states = [n.tag for n in nodes_by_level[level] if n.data]
            if len(states) > 0:
                if path_entropy is None:
                    path_entropy = 0.0
                path_entropy += self.shannon_entropy(states)
        return path_entropy

# policy = CausalEntropicPolicy((2, 2), max_sample_paths=10)
# s = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# print(s)
# tree = policy.all_paths(s, my_move=False)
# # tree.show()
# print(policy.causal_path_entropy(tree))
# tree = policy.all_paths(convert_edge_matrix_to_board_state([[1, 1, 1, 0, 1],
#                                                             [1, 0, 1, 0, 1],
#                                                             [1, 1, 1, 1, 1],
#                                                             [1, 0, 1, 0, 1],
#                                                             [1, 1, 1, 1, 1]]), my_move=True)
# tree.show()
# print(policy.causal_path_entropy(tree))
# tree = policy.all_paths(convert_edge_matrix_to_board_state([[1, 1, 1, 1, 1],
#                                                             [1, 0, 1, 0, 1],
#                                                             [1, 1, 1, 0, 1],
#                                                             [1, 0, 1, 0, 1],
#                                                             [1, 1, 1, 1, 1]]), my_move=False)
# tree.show()
# print(policy.causal_path_entropy(tree))

# policy = CausalEntropicPolicy((1, 1), max_sample_paths=10)
# tree = policy.all_paths([0, 0, 0, 0])
# print(policy.causal_path_entropy(tree))


import ast
import itertools
import random

from dnbpy import *


class TabularPolicy(Policy):
    def __init__(self, board_size, epsilon, initial_state_value, table_file_path=None, existing_value_table=None):
        """
        
        :param board_size: 
        :param epsilon: a scalar, from 0.0 to 1.0, that is the value used for epsilon-greedy action selection; a value
                        of 0.0 should be used for a completely greedy policy (i.e. a policy that always selects 
                        the move with highest value)
        :param table_file_path: 
        """
        self._board_size = board_size
        self._epsilon = epsilon
        if table_file_path is not None:
            self._value_table = self._init_value_table_from_file(table_file_path)
            self._symmetric_states = self._rebuild_symmetric_states(board_size)
        elif existing_value_table is not None:
            self._value_table = existing_value_table
            self._symmetric_states = self._rebuild_symmetric_states(board_size)
        else:
            # initialize value table, making sure not to repeat state symmetries
            self._value_table = {}
            self._symmetric_states = {}
            all_states = list(itertools.product([0, 1], repeat=len(init_board_state(board_size))))
            for state in all_states:
                edge_matrix = convert_board_state_to_edge_matrix(board_size, state)
                symmetries = convert_edge_matrix_to_symmetries(edge_matrix)
                all_symmetries_as_strings = []
                self._symmetric_states[as_string(state)] = all_symmetries_as_strings
                state_exists = False
                for symmetry in symmetries:
                    state_string = as_string(convert_edge_matrix_to_board_state(symmetry))
                    all_symmetries_as_strings.append(state_string)
                    if state_string in self._value_table:
                        state_exists = True
                if not state_exists:
                    self._value_table[as_string(state)] = initial_state_value() if callable(initial_state_value) else initial_state_value

    def _init_value_table_from_file(self, table_file_path):
        f = open(table_file_path, 'r')
        contents = f.read()
        return ast.literal_eval(contents)

    def _rebuild_symmetric_states(self, board_size):
        symmetric_states = {}
        all_states = list(itertools.product([0, 1], repeat=len(init_board_state(board_size))))
        for state in all_states:
            edge_matrix = convert_board_state_to_edge_matrix(board_size, state)
            symmetries = convert_edge_matrix_to_symmetries(edge_matrix)
            all_symmetries_as_strings = []
            symmetric_states[as_string(state)] = all_symmetries_as_strings
            for symmetry in symmetries:
                state_string = as_string(convert_edge_matrix_to_board_state(symmetry))
                all_symmetries_as_strings.append(state_string)
        return symmetric_states

    def _find_state_string(self, state):
        state_string = None
        for symmetry_string in self._symmetric_states[as_string(state)]:
            if symmetry_string in self._value_table:
                state_string = symmetry_string
                break
        if state_string is None:
            raise Exception("could not find selected state in value table: %s" % str(state))
        return state_string

    def select_edge(self, board_state):
        zero_indices = []
        for i in range(len(board_state)):
            if board_state[i] == 0:
                zero_indices.append(i)
        if random.random() < self._epsilon:
            return random.choice(zero_indices)
        else:
            best_value = 0.0
            best_state_index = zero_indices[0]
            for zero_index in zero_indices:
                new_state = [x for x in board_state]
                new_state[zero_index] = 1
                new_state_string = self._find_state_string(new_state)
                new_state_value = self._value_table[new_state_string]
                if new_state_value > best_value:
                    best_value = new_state_value
                    best_state_index = zero_index
            return best_state_index

    def get_value_table(self):
        return dict(self._value_table)

    def get_epsilon(self):
        return self._epsilon

    def set_epsilon(self, eps):
        self._epsilon = eps

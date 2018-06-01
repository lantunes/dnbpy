from ai import *


class TDZeroPolicy(TabularPolicy):
    def __init__(self, board_size, learning_rate=0.0, gamma=0.0, epsilon=0.0, initial_state_value=0.0, table_file_path=None):
        super(TDZeroPolicy, self).__init__(board_size=board_size, epsilon=epsilon,
                                           initial_state_value=initial_state_value, table_file_path=table_file_path)
        self._learning_rate = learning_rate
        self._gamma = gamma

    def update_value(self, reward, initial_state, selected_state):
        initial_state_string = self._find_state_string(initial_state)
        selected_state_string = self._find_state_string(selected_state)
        initial_state_value = self._value_table[initial_state_string]
        selected_state_value = self._value_table[selected_state_string]
        self._value_table[initial_state_string] = initial_state_value + self._learning_rate * (reward + (self._gamma * selected_state_value) - initial_state_value)

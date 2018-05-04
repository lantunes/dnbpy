from ai import *


class TDOnePolicy(TabularPolicy):
    def __init__(self, board_size, epsilon, learning_rate, gamma, initial_state_value=0.0, table_file_path=None):
        super(TDOnePolicy, self).__init__(board_size, epsilon, initial_state_value, table_file_path)
        self._learning_rate = learning_rate
        self._gamma = gamma

    def update_value(self, reward, backups):
        for i in reversed(range(len(backups))):
            state_string = self._find_state_string(backups[i])
            value = self._value_table[state_string]
            if i == len(backups) - 1:
                next_value = 0
                r = reward
            else:
                next_value = self._get_value(backups[i + 1])
                r = 0
            self._value_table[state_string] = value + self._learning_rate * (r + self._gamma * next_value - value)

    def get_learning_rate(self):
        return self._learning_rate

    def set_learning_rate(self, lr):
        self._learning_rate = lr

    def _get_value(self, state):
        state_string = self._find_state_string(state)
        return self._value_table[state_string]

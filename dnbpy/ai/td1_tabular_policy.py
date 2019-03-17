from dnbpy.ai import *


class TDOneTabularPolicy(TabularPolicy):
    def __init__(self, board_size, learning_rate=0.0, gamma=0.0, epsilon=0.0, initial_state_value=0.0,
                 table_file_path=None, softmax_action=False, temperature=0.0):
        super(TDOneTabularPolicy, self).__init__(board_size=board_size, epsilon=epsilon,
                                                 initial_state_value=initial_state_value, table_file_path=table_file_path,
                                                 softmax_action=softmax_action, temperature=temperature)
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

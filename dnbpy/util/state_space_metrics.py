from dnbpy import *
import itertools
import collections
import numpy as np
import matplotlib.pyplot as plt


class StateSpaceMetrics:
    def __init__(self, board_size):
        all_states = list(itertools.product([0, 1], repeat=len(init_board_state(board_size))))
        self._state_visitation_count = {}
        for state in all_states:
            self._state_visitation_count[''.join([str(x) for x in state])] = 0

    def get_state_visitation_counts(self):
        return collections.OrderedDict(sorted(self._state_visitation_count.items()))

    def get_state_visitation_count_matrix(self, shape):
        return np.reshape(list(self.get_state_visitation_counts().values()), shape)

    def state_visited(self, state):
        self._state_visitation_count[state] += 1

    def plot_state_visitation_count_heatmap(self, shape):
        plt.imshow(self.get_state_visitation_count_matrix(shape), cmap='tab20c_r', interpolation='nearest')
        plt.colorbar()
        plt.show()

# s = StateSpaceMetrics((2,2))
# # print(s.get_state_visitation_counts())
# s.state_visited('011000001111')
# s.plot_state_visitation_count_heatmap([64, 64])
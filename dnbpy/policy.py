import random


class Policy:
    def select_edge(self, board_state):
        raise NotImplementedError


class RandomPolicy(Policy):
    def select_edge(self, board_state):
        zero_indices = []
        for i in range(len(board_state)):
            if board_state[i] == 0:
                zero_indices.append(i)
        if len(zero_indices) == 0:
            raise Exception("there are no edges to select")
        return random.choice(zero_indices)

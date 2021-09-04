from random import Random
from .policy import Policy


class RandomPolicy(Policy):
    """
    Selects an edge randomly.
    """
    def __init__(self, random_state=None):
        """
        Construct a RandomPolicy instance.

        :param random_state: an optional random seed to use
        """
        self._random = Random(random_state)

    def select_edge(self, board_state, score=None, opp_score=None):
        zero_indices = []
        for i in range(len(board_state)):
            if board_state[i] == 0:
                zero_indices.append(i)
        if len(zero_indices) == 0:
            raise Exception("there are no edges to select")
        return self._random.choice(zero_indices)

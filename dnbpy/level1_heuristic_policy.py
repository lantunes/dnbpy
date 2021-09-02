from .policy import Policy
from .misc_functions import init_boxes
from random import Random


class Level1HeuristicPolicy(Policy):
    """
    Selects an edge randomly that completes a box, if possible. Otherwise, it selects an edge at random.
    """
    def __init__(self, board_size, random_state=None):
        """
        Construct an instance of Level1HeuristicPolicy.

        :param board_size: a 2-tuple representing the board size, i.e. (rows, columns)
        :param random_state: an optional random seed to use
        """
        self._board_size = board_size
        self._random = Random(random_state)

    def select_edge(self, board_state):
        boxes = init_boxes(board_state)
        complete_boxes = []
        for edge_index in range(len(board_state)):
            if board_state[edge_index] == 1:
                for box in boxes:
                    if box.contains(edge_index) and box.is_complete(board_state):
                        complete_boxes.append(box)
        zero_indices = []
        for i in range(len(board_state)):
            if board_state[i] == 0:
                zero_indices.append(i)
        self._random.shuffle(zero_indices)
        for zero_index in zero_indices:
            new_state = [x for x in board_state]
            new_state[zero_index] = 1
            for box in boxes:
                if box.contains(zero_index) and box.is_complete(new_state) and box not in complete_boxes:
                    return zero_index
        return self._random.choice(zero_indices)

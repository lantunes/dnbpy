from dnbpy import *
import random


class Level2HeuristicPolicy(Policy):
    """
    Selects an edge randomly that completes a box, if possible. Otherwise, it selects an edge randomly that does not 
     allow the opponent to complete a box. Otherwise, it selects an edge at random.
    """
    def __init__(self, board_size):
        self._board_size = board_size

    def select_edge(self, board_state, tensor_state=None):
        selected_edge = self.select_edge_completing_box(board_state)
        if selected_edge != -1:
            return selected_edge
        zero_indices = []
        for i in range(len(board_state)):
            if board_state[i] == 0:
                zero_indices.append(i)
        random.shuffle(zero_indices)
        for zero_index in zero_indices:
            new_state = [x for x in board_state]
            new_state[zero_index] = 1
            # check if opponent could complete a box if this edge were selected
            opponent_edge = self.select_edge_completing_box(new_state)
            if opponent_edge == -1:
                # the opponent could not make a box if this edge were selected
                return zero_index
        # there are no edges that could be selected that would prevent the opponent from making a box;
        #  pick an edge at random
        return random.choice(zero_indices)

    def select_edge_completing_box(self, board_state):
        boxes = self._init_boxes()
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
        random.shuffle(zero_indices)
        for zero_index in zero_indices:
            new_state = [x for x in board_state]
            new_state[zero_index] = 1
            for box in boxes:
                if box.contains(zero_index) and box.is_complete(new_state) and box not in complete_boxes:
                    return zero_index
        return -1

    def _init_boxes(self):
        boxes = []
        rows = self._board_size[0]
        cols = self._board_size[1]
        r_index = 0
        for r in range(rows):
            for c in range(cols):
                boxes.append(Box(r_index, r_index + cols, r_index + cols + 1, 2*cols + 1 + r_index,r,c))
                r_index += 1
            r_index += cols + 1
        return boxes
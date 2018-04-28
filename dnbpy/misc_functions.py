import numpy as np


def convert_vector_index_to_coordinates(board_size, vector_index):
    """
    Converts vector index to edge matrix coordinates
    :param board_size: a tuple of ints, (rows, cols), representing the board dimensions
    :param vector_index: an int representing the index within the vector representation of the board state
    :return: a tuple, (x, y), representing the coordinates of the edge in the edge matrix
    """
    cols = board_size[1]
    x = 0
    y = 0
    next_x = 0
    for i in range(vector_index + 1):
        x = next_x
        if x % 2 == 0:
            if y == 2*cols:
                y = 1
            else:
                y += 1 if y == 0 else 2
            if y == (2*cols - 1):
                next_x += 1
        else:
            y = 0 if y == (2*cols - 1) else y + 2
            if y == 2*cols:
                next_x += 1
    return x, y


def convert_edge_matrix_to_board_state(edge_matrix):
    edge_matrix = np.array(edge_matrix)
    rows = edge_matrix.shape[0] // 2
    cols = edge_matrix.shape[1] // 2
    board_state = [0]*((2*rows*cols) + rows + cols)
    for i in range(len(board_state)):
        coordinates = convert_vector_index_to_coordinates((rows, cols), i)
        board_state[i] = edge_matrix[coordinates]
    return board_state

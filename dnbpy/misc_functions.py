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


def as_string(board_state):
    return ''.join([str(x) for x in board_state])


def init_board_state(board_size):
    rows = board_size[0]
    cols = board_size[1]
    return [0]*((2*rows*cols) + rows + cols)


def init_edge_matrix(board_size):
    rows = board_size[0]
    cols = board_size[1]
    even = [1 if i % 2 == 0 else 0 for i in range((2*cols + 1))]
    odd = [0] * (2*cols + 1)
    return np.array([even if i % 2 == 0 else odd for i in range((2*rows + 1))])


def convert_board_state_to_edge_matrix(board_size, board_state):
    edge_matrix = init_edge_matrix(board_size)
    for i in range(len(board_state)):
        coordinates = convert_vector_index_to_coordinates(board_size, i)
        edge_matrix[coordinates] = board_state[i]
    return edge_matrix.tolist()


def convert_edge_matrix_to_board_state(edge_matrix):
    edge_matrix = np.array(edge_matrix)
    rows = edge_matrix.shape[0] // 2
    cols = edge_matrix.shape[1] // 2
    board_state = [0]*((2*rows*cols) + rows + cols)
    for i in range(len(board_state)):
        coordinates = convert_vector_index_to_coordinates((rows, cols), i)
        board_state[i] = edge_matrix[coordinates]
    return board_state


def convert_edge_matrix_to_symmetries(edge_matrix):
    # I symmetry
    i = np.array(edge_matrix)
    rows = i.shape[0]
    cols = i.shape[1]

    if rows == cols:
        # S symmetry
        s = np.flip(i, 1)
        # R symmetry
        r = np.rot90(i, 1)
        # R^2 symmetry
        r2 = np.rot90(i, 2)
        # R^3 symmetry
        r3 = np.rot90(i, 3)
        # SR symmetry
        sr = np.rot90(s, 1)
        # SR^2 symmetry
        sr2 = np.rot90(s, 2)
        # SR^3 symmetry
        sr3 = np.rot90(s, 3)
        all_possible = [i, s, r, r2, r3, sr, sr2, sr3]
    else:
        # S symmetry
        s = np.flip(i, 1)
        # R^2 symmetry
        r2 = np.rot90(i, 2)
        # SR^2 symmetry
        sr2 = np.rot90(s, 2)
        all_possible = [i, s, r2, sr2]

    def contains(list, arr):
        for item in list:
            if np.array_equal(item, arr):
                return True
        return False

    symmetries = []
    for sym in all_possible:
        if not contains(symmetries, sym):
            symmetries.append(sym.tolist())

    return symmetries

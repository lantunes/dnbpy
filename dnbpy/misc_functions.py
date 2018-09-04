import numpy as np


def convert_vector_index_to_coordinates(board_size, vector_index, edge_length=1, include_dots=False):
    cols = board_size[1]
    row_index = (vector_index//(2*cols+1)) * (edge_length + 1)
    normalized_vector_index = vector_index - (vector_index//(2*cols+1))*(2*cols+1)
    is_horizontal_edge = normalized_vector_index < cols
    if is_horizontal_edge:
        col_index = normalized_vector_index * (edge_length + 1)
        start_dot_coords = (row_index, col_index)
        edge_coords = [(row_index, col_index + e + 1) for e in range(edge_length)]
        end_dot_coords = (row_index, col_index + edge_length + 1)
    else:
        col_index = (normalized_vector_index - cols) * (edge_length + 1)
        start_dot_coords = (row_index, col_index)
        edge_coords = [(row_index + e + 1, col_index) for e in range(edge_length)]
        end_dot_coords = (row_index + edge_length + 1, col_index)
    final_coords = []
    if include_dots:
        final_coords.append(start_dot_coords)
    final_coords.extend(edge_coords)
    if include_dots:
        final_coords.append(end_dot_coords)
    return final_coords


def as_string(board_state):
    return ''.join([str(x) for x in board_state])


def init_board_state(board_size):
    rows = board_size[0]
    cols = board_size[1]
    return [0]*((2*rows*cols) + rows + cols)


def init_edge_matrix(board_size, edge_length=1, include_dots=True):
    rows = board_size[0]
    cols = board_size[1]
    width = (cols + 1) + (cols * edge_length)
    if include_dots:
        even = [1.0]
        for i in range(cols):
            for j in range(edge_length):
                even.append(0.0)
            even.append(1.0)
    else:
        even = [0.0] * width
    odd = [0.0] * width
    edge_matrix = [even]
    for i in range(rows):
        for j in range(edge_length):
            edge_matrix.append(odd)
        edge_matrix.append(even)
    return np.array(edge_matrix)


def convert_board_state_to_edge_matrix(board_size, board_state, edge_length=1, include_dots=True):
    edge_matrix = init_edge_matrix(board_size, edge_length, include_dots)
    for i in range(len(board_state)):
        coordinates = convert_vector_index_to_coordinates(board_size, i, edge_length, include_dots=not include_dots)
        for coord in coordinates:
            if edge_matrix[coord] != 1:
                edge_matrix[coord] = board_state[i]
    return edge_matrix.tolist()


def convert_edge_matrix_to_board_state(edge_matrix, edge_length=1):
    edge_matrix = np.array(edge_matrix)
    rows = edge_matrix.shape[0] // (2 + (edge_length - 1))
    cols = edge_matrix.shape[1] // (2 + (edge_length - 1))
    board_state = [0]*((2*rows*cols) + rows + cols)
    for i in range(len(board_state)):
        coordinates = convert_vector_index_to_coordinates((rows, cols), i, edge_length, include_dots=False)
        for coord in coordinates:
            board_state[i] = edge_matrix[coord]
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

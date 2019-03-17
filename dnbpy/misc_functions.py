
def as_string(board_state):
    return ''.join([str(x) for x in board_state])


def init_board_state(board_size):
    rows = board_size[0]
    cols = board_size[1]
    return [0]*((2*rows*cols) + rows + cols)



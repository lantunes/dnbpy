from .box import Box


def as_string(board_state):
    return ''.join([str(x) for x in board_state])


def init_board_state(board_size):
    rows = board_size[0]
    cols = board_size[1]
    return [0]*((2*rows*cols) + rows + cols)


def init_boxes(board_size):
    boxes = []
    rows = board_size[0]
    cols = board_size[1]
    r_index = 0
    for r in range(rows):
        for c in range(cols):
            boxes.append(Box(r_index, r_index + cols, r_index + cols + 1, 2*cols + 1 + r_index))
            r_index += 1
        r_index += cols + 1
    return boxes

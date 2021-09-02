
class Box:
    def __init__(self, edge1, edge2, edge3, edge4):
        self._edges = [edge1, edge2, edge3, edge4]

    def contains(self, edge):
        return edge in self._edges

    def get_edges(self):
        return [edge for edge in self._edges]

    def is_complete(self, board_state):
        return board_state[self._edges[0]] == 1 and board_state[self._edges[1]] == 1 and \
               board_state[self._edges[2]] == 1 and board_state[self._edges[3]] == 1

    def __str__(self):
        return '-'.join([str(x) for x in self._edges])

    def __eq__(self, other):
        if not isinstance(other, Box):
            return False
        return self._edges == other._edges

    def __hash__(self):
        return hash(tuple(self._edges))

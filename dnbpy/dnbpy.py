class GameEngine:
    def __init__(self, board_size, players):
        if not board_size:
            raise Exception("board size must be specified")
        if not isinstance(board_size, tuple):
            raise Exception("board size must be a tuple")
        if len(board_size) != 2:
            raise Exception("number of rows and columns must be specified")
        if board_size[0] <= 0 or not isinstance(board_size[0], int):
            raise Exception("invalid board size: %s" % ','.join(board_size))
        if board_size[1] <= 0 or not isinstance(board_size[1], int):
            raise Exception("invalid board size: %s" % ','.join(board_size))
        if len(players) < 2:
            raise Exception("there must be at least two players specified")
        self._board_size = board_size
        self._players = players
        self._current_player = 0
        rows = board_size[0]
        cols = board_size[1]
        self._board_state = [0]*((2*rows*cols) + rows + cols)
        self._boxes = self._init_boxes()
        self._players_to_boxes = {}
        for player in players:
            self._players_to_boxes[player] = []

    def _init_boxes(self):
        boxes = []
        rows = self._board_size[0]
        cols = self._board_size[1]
        r_index = 0
        for r in range(rows):
            for c in range(cols):
                boxes.append(Box(r_index, r_index + cols, r_index + cols + 1, 2*cols + 1 + r_index))
                r_index += 1
            r_index += cols + 1
        return boxes

    def get_board_state(self):
        return [s for s in self._board_state]

    def select_edge(self, edge_index, player):
        """
        Selects an edge on the game board. 
        :param edge_index: the index of the edge to select 
        :param player: the player selecting the edge
        :return: the next player to play
        """
        if edge_index < 0 or edge_index > len(self._board_state) - 1:
            raise Exception("invalid edge index: %s" % edge_index)
        if self._board_state[edge_index] == 1:
            raise Exception("edge already selected: %s" % edge_index)
        if player != self._players[self._current_player]:
            raise Exception("next player to play is %s" % self._players[self._current_player])
        self._board_state[edge_index] = 1
        boxes_made = 0
        for box in self._boxes:
            if box.contains(edge_index) and box.is_complete(self._board_state) and box not in self._players_to_boxes[player]:
                self._players_to_boxes[player].append(box)
                boxes_made += 1
        if boxes_made == 0:
            self._current_player = (self._current_player + 1) % len(self._players)
        return self._players[self._current_player]

    def get_score(self, player):
        if player not in self._players:
            raise Exception("player not recognized: %s" % player)
        return len(self._players_to_boxes[player])

    def get_current_player(self):
        return self._players[self._current_player]

    def get_boxes(self, player):
        if player not in self._players:
            raise Exception("player not recognized: %s" % player)
        return self._players_to_boxes[player]

    def get_all_boxes(self):
        return self._boxes


class Box:
    def __init__(self, edge1, edge2, edge3, edge4):
        self._edges = [edge1, edge2, edge3, edge4]

    def contains(self, edge):
        return edge in self._edges

    def is_complete(self, board_state):
        return board_state[self._edges[0]] == 1 and board_state[self._edges[1]] == 1 and \
               board_state[self._edges[2]] == 1 and board_state[self._edges[3]] == 1

    def __str__(self):
        return '-'.join([str(x) for x in self._edges])

import numpy as np


class Game:
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
        self._edge_matrix = self._init_edge_matrix()
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

    def _init_edge_matrix(self):
        rows = self._board_size[0]
        cols = self._board_size[1]
        even = [1 if i % 2 == 0 else 0 for i in range((2*cols + 1))]
        odd = [0] * (2*cols + 1)
        return np.array([even if i % 2 == 0 else odd for i in range((2*rows + 1))])

    def get_board_size(self):
        return self._board_size[0], self._board_size[1]

    def get_players(self):
        return [player for player in self._players]

    def convert_vector_index_to_coordinates(self, vector_index):
        """
        Converts vector index to edge matrix coordinates
        :param vector_index: an int representing the index within the vector representation of the board state
        :return: a tuple, (x, y), representing the coordinates of the edge in the edge matrix
        """
        cols = self._board_size[1]
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

    def convert_edge_matrix_to_board_state(self, edge_matrix):
        edge_matrix = np.array(edge_matrix)
        rows = edge_matrix.shape[0] // 2
        cols = edge_matrix.shape[1] // 2
        board_state = [0]*((2*rows*cols) + rows + cols)
        for i in range(len(board_state)):
            coordinates = self.convert_vector_index_to_coordinates(i)
            board_state[i] = edge_matrix[coordinates]
        return board_state

    def get_board_state(self):
        """
        The board edges are indexed as follows (for the 2x2 case):
        
        *  0  *  1  * 
        2     3     4
        *  5  *  6  *
        7     8     9
        *  10 *  11 *
        
        In the example above, if edges 1 and 5 are selected, the board state will be [0,1,0,0,0,1,0,0,0,0,0].
        :return: a binary list representing the board state 
        """
        return [s for s in self._board_state]

    def select_edge(self, edge_index, player):
        """
        Selects an edge on the game board. 
        :param edge_index: the index of the edge to select 
        :param player: the player selecting the edge
        :return: the next player to play, or None if the game is finished
        """
        if self.is_finished():
            raise Exception("game is finished")
        if edge_index < 0 or edge_index > len(self._board_state) - 1:
            raise Exception("invalid edge index: %s" % edge_index)
        if self._board_state[edge_index] == 1:
            raise Exception("edge already selected: %s" % edge_index)
        if player != self._players[self._current_player]:
            raise Exception("next player to play is: %s" % self._players[self._current_player])
        self._board_state[edge_index] = 1
        # also, update status of the matrix
        coordinates = self.convert_vector_index_to_coordinates(edge_index)
        self._edge_matrix[coordinates] += 1
        boxes_made = 0
        for box in self._boxes:
            if box.contains(edge_index) and box.is_complete(self._board_state) and box not in self._players_to_boxes[player]:
                self._players_to_boxes[player].append(box)
                boxes_made += 1
        if boxes_made == 0:
            self._current_player = (self._current_player + 1) % len(self._players)
        return self.get_current_player()

    def get_score(self, player):
        if player not in self._players:
            raise Exception("player not recognized: %s" % player)
        return len(self._players_to_boxes[player])

    def get_current_player(self):
        """
        Returns the current player to play, or None if the game is finished.
        :return: the current player to play, or None if the game is finished 
        """
        return None if self.is_finished() else self._players[self._current_player]

    def get_boxes(self, player):
        if player not in self._players:
            raise Exception("player not recognized: %s" % player)
        return self._players_to_boxes[player]

    def get_all_boxes(self):
        return [box for box in self._boxes]

    def is_finished(self):
        """
        Whether the game is finished. The game is finished when all boxes are completed.
        :return: whether the game is finished
        """
        return sum(self._board_state) == len(self._board_state)

    def get_edge_matrix(self):
        return np.array(self._edge_matrix)

    def __str__(self):
        return ToString().apply(self)


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


class ToString:
    def __init__(self):
        pass

    def apply(self, game):
        rows = game.get_board_size()[0]
        cols = game.get_board_size()[1]
        board_state = game.get_board_state()
        edge_to_boxes = self._get_edge_to_boxes(game.get_all_boxes())
        players = game.get_players()
        cur_index = 0
        board = ""
        for row in range(rows):
            if row == 0:
                for c, col in enumerate(range(cols)):
                    edge_val = "-" if self._is_edge_selected(board_state, cur_index) else cur_index
                    if c == 0:
                        board += '*    {0: <3}  *'.format(edge_val)
                    else:
                        board += '    {0: <3}  *'.format(edge_val)
                    cur_index += 1
                board += '  \n                                 \n'
            for c, col in enumerate(range(cols + 1)):
                edge_val = "|" if self._is_edge_selected(board_state, cur_index) else cur_index
                board += '{0: <3}'.format(edge_val)
                if c < cols:
                    player_for_box = self._get_player_for_box(cur_index, edge_to_boxes, game, players)
                    if player_for_box is not None:
                        board += ' {0: <5.5} '.format(player_for_box)
                    else:
                        board += '       '
                cur_index += 1
            board += '\n                                 \n'
            for c, col in enumerate(range(cols)):
                edge_val = "-" if self._is_edge_selected(board_state, cur_index) else cur_index
                if c == 0:
                    board += '*    {0: <3}  *'.format(edge_val)
                else:
                    board += '    {0: <3}  *'.format(edge_val)
                cur_index += 1
            board += '  \n                                 \n'
        board += self._print_player_scores(game, players)
        return board

    def _is_edge_selected(self, board_state, edge_index):
        return board_state[edge_index] == 1

    def _get_edge_to_boxes(self, boxes):
        edge_to_boxes = {}
        for box in boxes:
            edge_to_boxes[box.get_edges()[1]] = box
        return edge_to_boxes

    def _get_player_for_box(self, edge, edge_to_boxes, game, players):
        box = edge_to_boxes[edge]
        for player in players:
            player_boxes = game.get_boxes(player)
            for player_box in player_boxes:
                if str(box) == str(player_box):
                    return player
        return None

    def _print_player_scores(self, game, players):
        scores = ""
        for i, player in enumerate(players):
            scores += "{0:}: {1:}".format(player, game.get_score(player))
            if i < len(players) - 1:
                scores += ", "
        return scores

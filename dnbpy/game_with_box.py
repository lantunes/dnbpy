from .misc_functions import *
from .to_string import *


class Game_With_Box:
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
        self._board_state = init_board_state(board_size)
        self._boxes = self._init_boxes()
        self._players_to_boxes = {}
        self._box_matrix = {}

        self._edge_matrix = init_edge_matrix(board_size)

        for player in players: self._box_matrix[player] = init_box_matrix(board_size)

        for player in players:
            self._players_to_boxes[player] = []

    def _init_boxes(self):
        boxes = []
        rows = self._board_size[0]
        cols = self._board_size[1]
        r_index = 0
        for r in range(rows):
            for c in range(cols):
                boxes.append(Box(r_index, r_index + cols, r_index + cols + 1, 2 * cols + 1 + r_index, r, c))
                r_index += 1
            r_index += cols + 1
        return boxes

    def get_board_size(self):
        return self._board_size[0], self._board_size[1]

    def get_players(self):
        return [player for player in self._players]

    def get_board_box_matrix(self):
        return (self._box_matrix)

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
        :return: the next player to play, or None if the game is finished, and the number of boxes made
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
        coordinates = convert_vector_index_to_coordinates(self._board_size, edge_index)
        self._edge_matrix[coordinates] += 1
        boxes_made = 0
        for box in self._boxes:
            # if box.is_complete(self._board_state):
            #   row_index,col_index = box.get_indices()
            #  self._box_matrix[player][2*row_index+1,2*col_index+1] = 1 #Map of completed boxes for each player

            if box.contains(edge_index) and box.is_complete(self._board_state) and box not in self._players_to_boxes[
                player]:
                row_index, col_index = box.get_indices()
                self._box_matrix[player][
                    2 * row_index + 1, 2 * col_index + 1] = 1  # Map of completed boxes for each player
                self._players_to_boxes[player].append(box)
                boxes_made += 1
        if boxes_made == 0:
            self._current_player = (self._current_player + 1) % len(self._players)
        return self.get_current_player(), boxes_made

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

    def get_tensor_representation(self,current_player,num_channels):

        out_tensor = []
        #reshaped_edge_matrix = np.reshape(self.get_edge_matrix(),(1,)+np.shape(self.get_edge_matrix())+(1,))
        out_tensor.append(self.get_edge_matrix())
        #reshaped_box_matrix = np.reshape(np.array(self._box_matrix[current_player]),(1,)+np.shape(np.array(self._box_matrix[current_player]))+(1,))
        if num_channels==2:
            out_tensor.append(np.array(self._box_matrix[current_player]))
        elif num_channels==4:
            out_tensor.append(np.array(self._box_matrix[current_player]))
            if current_player==self._players[0]:
               out_tensor.append(np.array(self._box_matrix[self._players[1]]))
            else:
               out_tensor.append(np.array(self._box_matrix[self._players[0]]))
            shape = np.shape(self.get_edge_matrix())
            out_tensor.append(np.ones([shape[0],shape[1]])*self._current_player)
        return (out_tensor)


    def __str__(self):
        return ToString().apply(self)


class Box:
    def __init__(self, edge1, edge2, edge3, edge4, row_index, column_index):
        self._edges = [edge1, edge2, edge3, edge4]
        self._row_index = row_index
        self._col_index = column_index

    def contains(self, edge):
        return edge in self._edges

    def get_edges(self):
        return [edge for edge in self._edges]

    def get_indices(self):
        return (self._row_index, self._col_index)

    def is_complete(self, board_state):
        return board_state[self._edges[0]] == 1 and board_state[self._edges[1]] == 1 and \
               board_state[self._edges[2]] == 1 and board_state[self._edges[3]] == 1

    def __str__(self):
        return '-'.join([str(x) for x in self._edges])
from .misc_functions import *
from .to_string import *


class Game:
    def __init__(self, board_size, players, board_state=None):
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
        self._board_state = [s for s in board_state] if board_state is not None else init_board_state(board_size)
        if board_state is not None and self.is_finished():
            raise Exception("game is already finished")
        self._boxes = init_boxes(board_size)
        self._players_to_boxes = {}
        for player in players:
            self._players_to_boxes[player] = set()

    def get_board_size(self):
        return self._board_size[0], self._board_size[1]

    def get_players(self):
        return [player for player in self._players]

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
        boxes_made = 0
        for box in self._boxes:
            if box.contains(edge_index) and box.is_complete(self._board_state) and box not in self._players_to_boxes[player]:
                self._players_to_boxes[player].add(box)
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
        return list(self._players_to_boxes[player])

    def get_all_boxes(self):
        return [box for box in self._boxes]

    def get_legal_moves(self):
        zero_indices = [i for i, v in enumerate(self._board_state) if v == 0]
        return zero_indices

    def is_finished(self):
        """
        Whether the game is finished. The game is finished when all boxes are completed.
        :return: whether the game is finished
        """
        return sum(self._board_state) == len(self._board_state)

    def __str__(self):
        return ToString().apply(self)

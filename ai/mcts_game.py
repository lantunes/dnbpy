from dnbpy import *


class MCTSGame:
    """
    A game engine used for MCTS playouts. Currently supports only 2 players. The first player is assumed to be
    the root player.
    """
    def __init__(self, board_size, players, board_state=None, root_player_score=None):
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
        if len(players) != 2:
            raise Exception("there must be exactly two players specified")
        self._board_size = board_size
        self._players = players
        self._current_player = 0
        self._board_state = [s for s in board_state] if board_state is not None else init_board_state(board_size)
        if board_state is not None and self.is_finished():
            raise Exception("game is already finished")
        self._root_player_score = root_player_score if root_player_score is not None else 0
        # we don't need to store the opponent score as that will be determined at the end of the game by subtracting the
        #  root player's score from the total number of boxes on the board
        self._boxes = self._init_boxes()

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

    def select_edge(self, edge_index, player):
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
            if box.contains(edge_index) and box.is_complete(self._board_state):
                boxes_made += 1
        if self._is_root(player):
            self._root_player_score += boxes_made
        if boxes_made == 0:
            self._current_player = (self._current_player + 1) % len(self._players)
        return self.get_current_player()

    def get_current_player(self):
        return None if self.is_finished() else self._players[self._current_player]

    def is_finished(self):
        return sum(self._board_state) == len(self._board_state)

    def get_score(self, player):
        if not self.is_finished():
            raise Exception("player scores are available only at the end of the game")
        if player not in self._players:
            raise Exception("player not recognized: %s" % player)
        if self._is_root(player):
            return self._root_player_score
        else:
            return (self._board_size[0] * self._board_size[1]) - self._root_player_score

    def _is_root(self, player):
        return player == self._players[0]

    def get_board_state(self):
        return [s for s in self._board_state]

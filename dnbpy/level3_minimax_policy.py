from .policy import Policy
import math
from random import Random
from .game import Game


class Level3MinimaxPolicy(Policy):
    """
    A depth-limited minimax policy with alpha-beta pruning. A random edge is selected if there is no best edge.
    The depth parameter can be either an int or a callable, if the depth is to be determined dynamically given the
    board state. If depth is None, then a built-in callable depth is used.

    Performance in duels vs. L2:
    depth=3,    update_alpha=False: won 38/48 in ~166 sec
    depth=3,    update_alpha=True:  won 35/48 in ~20 sec
    depth=None, update_alpha=False: won 41/48 in ~175 sec
    depth=None, update_alpha=True:  won 45/48 in ~74 sec

    See https://en.wikipedia.org/wiki/Minimax and
    https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning for more information.
    """
    def __init__(self, board_size, depth=None, random_state=None, heuristic=None, update_alpha=False):
        self._board_size = board_size
        self._depth = depth
        if self._depth is None:
            self._depth = self._get_depth
        self._random = Random(random_state)
        self._heuristic = heuristic
        self._update_alpha = update_alpha

    def select_edge(self, board_state, score=None, opp_score=None):
        game = Game(self._board_size, players=["max", "min"], board_state=board_state)
        edges = []
        values = []
        depth = self._depth(board_state) if callable(self._depth) else self._depth
        alpha = -math.inf
        for edge in game.get_legal_moves():
            game = Game(self._board_size, players=["max", "min"], board_state=board_state)
            next_player, boxes = game.select_edge(edge, "max")
            value = self._minimax(game, depth, alpha, math.inf, next_player, score+boxes, opp_score)
            if self._update_alpha:
                alpha = max(value, alpha)
            edges.append(edge)
            values.append(value)
        if min(values) == max(values):
            # there is no best edge; select randomly
            return self._random.choice(edges)
        return edges[values.index(max(values))]

    def _minimax(self, game, depth, alpha, beta, player, max_score, min_score):
        if depth == 0 or game.is_finished():
            return self._eval(game.get_board_state(), max_score, min_score, player == "max")

        if player == "max":
            value = -math.inf
            for edge in game.get_legal_moves():
                new_game = Game(self._board_size, players=["max", "min"], board_state=game.get_board_state())
                next_player, boxes = new_game.select_edge(edge, player)
                value = max(value, self._minimax(new_game, depth-1, alpha, beta, next_player, max_score+boxes, min_score))
                if value >= beta:
                    break  # beta cutoff
                alpha = max(alpha, value)
            return value

        else:  # minimizing player
            value = math.inf
            for edge in game.get_legal_moves():
                new_game = Game(self._board_size, players=["min", "max"], board_state=game.get_board_state())
                next_player, boxes = new_game.select_edge(edge, player)
                value = min(value, self._minimax(new_game, depth-1, alpha, beta, next_player, max_score, min_score+boxes))
                if value <= alpha:
                    break  # alpha cutoff
                beta = min(beta, value)
            return value

    def _eval(self, board_state, max_score, min_score, is_max):
        if self._heuristic is not None:
            return self._heuristic(board_state, max_score, min_score, is_max)
        return max_score - min_score

    def _get_depth(self, board_state):
        k = 2.5  # ideal for 3x3
        if self._board_size[0] * self._board_size[1] > 9:
            k = 1.5  # ideal for 5x5
        return int(1.5 + math.e ** (sum(board_state) / (len(board_state) / k)))

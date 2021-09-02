from .policy import Policy
import math
from random import Random
from .game import Game


class Level3MinimaxPolicy(Policy):
    """
    A depth-limited minimax policy with alpha-beta pruning. A random edge is selected if there is no best edge.
    See https://en.wikipedia.org/wiki/Minimax and
    https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning for more information.
    """
    def __init__(self, board_size, depth, random_state=None):
        self._board_size = board_size
        self._depth = depth
        self._random = Random(random_state)

    def select_edge(self, board_state):
        game = Game(self._board_size, players=["max", "min"], board_state=board_state)
        edges = []
        values = []
        for edge in game.get_legal_moves():
            game = Game(self._board_size, players=["max", "min"], board_state=board_state)
            next_player, boxes = game.select_edge(edge, "max")
            max_score = boxes
            min_score = 0
            value = self._minimax(game, self._depth, -math.inf, math.inf, next_player, max_score, min_score)
            edges.append(edge)
            values.append(value)
        if min(values) == max(values):
            # there is no best edge; select randomly
            return self._random.choice(edges)
        return edges[values.index(max(values))]

    def _minimax(self, game, depth, alpha, beta, player, max_score, min_score):
        if depth == 0 or game.is_finished():
            return max_score - min_score

        if player == "max":
            value = -math.inf
            for edge in game.get_legal_moves():
                game = Game(self._board_size, players=["max", "min"], board_state=game.get_board_state())
                next_player, boxes = game.select_edge(edge, player)
                value = max(value, self._minimax(game, depth-1, alpha, beta, next_player, max_score+boxes, min_score))
                if value >= beta:
                    break  # beta cutoff
                alpha = max(alpha, value)
            return value

        else:  # minimizing player
            value = math.inf
            for edge in game.get_legal_moves():
                game = Game(self._board_size, players=["min", "max"], board_state=game.get_board_state())
                next_player, boxes = game.select_edge(edge, player)
                value = min(value, self._minimax(game, depth-1, alpha, beta, next_player, max_score, min_score+boxes))
                if value <= alpha:
                    break  # alpha cutoff
                beta = min(beta, value)
            return value

import unittest

import dnbpy


class TestLevel3MinimaxPolicy(unittest.TestCase):

    def test_policy(self):
        board_size = (3, 3)
        players = {
            "p1": dnbpy.Level3MinimaxPolicy(board_size, 3, random_state=0),
            "p2": dnbpy.Level2HeuristicPolicy(board_size, random_state=0)
        }

        game = dnbpy.Game(board_size, ["p1", "p2"])

        while not game.is_finished():
            current_player = game.get_current_player()
            edge = players[current_player].select_edge(game.get_board_state())
            game.select_edge(edge, current_player)

        self.assertEqual(9, game.get_score("p1"))
        self.assertEqual(0, game.get_score("p2"))

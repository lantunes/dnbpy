import unittest

import dnbpy


class TestMCTSPolicy(unittest.TestCase):

    def test_policy(self):
        board_size = (3, 3)
        players = {
            "p1": dnbpy.Level1HeuristicPolicy(board_size, random_state=0),
            "p2": dnbpy.MCTSPolicy(board_size, 500, reset_tree=True, random_state=0)
        }

        game = dnbpy.Game(board_size, ["p1", "p2"])

        while not game.is_finished():
            current_player = game.get_current_player()
            opp_player = "p2" if current_player == "p1" else "p1"
            edge = players[current_player].select_edge(
                game.get_board_state(), game.get_score(current_player), game.get_score(opp_player))
            game.select_edge(edge, current_player)

        self.assertEqual(4, game.get_score("p1"))
        self.assertEqual(5, game.get_score("p2"))

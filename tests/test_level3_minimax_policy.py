import unittest
import math
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
            opp_player = "p2" if current_player == "p1" else "p1"
            edge = players[current_player].select_edge(
                game.get_board_state(), game.get_score(current_player), game.get_score(opp_player))
            game.select_edge(edge, current_player)

        self.assertEqual(5, game.get_score("p1"))
        self.assertEqual(4, game.get_score("p2"))

    def test_policy_with_heuristic(self):
        board_size = (3, 3)

        def heuristic(board_state, max_score, min_score, is_max):
            return -min_score

        players = {
            "p1": dnbpy.Level3MinimaxPolicy(board_size, 3, random_state=0, heuristic=heuristic),
            "p2": dnbpy.Level2HeuristicPolicy(board_size, random_state=0)
        }

        game = dnbpy.Game(board_size, ["p1", "p2"])

        while not game.is_finished():
            current_player = game.get_current_player()
            opp_player = "p2" if current_player == "p1" else "p1"
            edge = players[current_player].select_edge(
                game.get_board_state(), game.get_score(current_player), game.get_score(opp_player))
            game.select_edge(edge, current_player)

        self.assertEqual(5, game.get_score("p1"))
        self.assertEqual(4, game.get_score("p2"))

    def test_policy_with_no_depth(self):
        board_size = (3, 3)

        players = {
            "p1": dnbpy.Level3MinimaxPolicy(board_size, random_state=0),
            "p2": dnbpy.Level2HeuristicPolicy(board_size, random_state=0)
        }

        game = dnbpy.Game(board_size, ["p1", "p2"])

        while not game.is_finished():
            current_player = game.get_current_player()
            opp_player = "p2" if current_player == "p1" else "p1"
            edge = players[current_player].select_edge(
                game.get_board_state(), game.get_score(current_player), game.get_score(opp_player))
            game.select_edge(edge, current_player)

        self.assertEqual(7, game.get_score("p1"))
        self.assertEqual(2, game.get_score("p2"))

    def test_policy_with_callable_depth(self):
        board_size = (3, 3)

        def depth(board_state):
            return int(1.5 + math.e ** (sum(board_state) / (len(board_state)/2.5)))

        players = {
            "p1": dnbpy.Level3MinimaxPolicy(board_size, depth, random_state=0),
            "p2": dnbpy.Level2HeuristicPolicy(board_size, random_state=0)
        }

        game = dnbpy.Game(board_size, ["p1", "p2"])

        while not game.is_finished():
            current_player = game.get_current_player()
            opp_player = "p2" if current_player == "p1" else "p1"
            edge = players[current_player].select_edge(
                game.get_board_state(), game.get_score(current_player), game.get_score(opp_player))
            game.select_edge(edge, current_player)

        self.assertEqual(7, game.get_score("p1"))
        self.assertEqual(2, game.get_score("p2"))

    def test_policy_update_alpha(self):
        board_size = (3, 3)
        players = {
            "p1": dnbpy.Level3MinimaxPolicy(board_size, random_state=0, update_alpha=True),
            "p2": dnbpy.Level2HeuristicPolicy(board_size, random_state=0)
        }

        game = dnbpy.Game(board_size, ["p1", "p2"])

        while not game.is_finished():
            current_player = game.get_current_player()
            opp_player = "p2" if current_player == "p1" else "p1"
            edge = players[current_player].select_edge(
                game.get_board_state(), game.get_score(current_player), game.get_score(opp_player))
            game.select_edge(edge, current_player)

        self.assertEqual(7, game.get_score("p1"))
        self.assertEqual(2, game.get_score("p2"))

import unittest

import dnbpy


class TestPolicyDuel(unittest.TestCase):

    def test_level1_vs_random(self):
        board_size = (2, 2)
        players = {
            0: dnbpy.RandomPolicy(random_state=0),
            1: dnbpy.Level1HeuristicPolicy(board_size, random_state=0)
        }
        result = dnbpy.duel(board_size, players)
        self.assertEqual({'won': 18, 'tied': 5, 'lost': 1}, result[1])

        board_size = (3, 3)
        players = {
            0: dnbpy.RandomPolicy(random_state=0),
            1: dnbpy.Level1HeuristicPolicy(board_size, random_state=0)
        }
        result = dnbpy.duel(board_size, players)
        self.assertEqual({'lost': 0, 'tied': 0, 'won': 48}, result[1])

    def test_level1_vs_level2(self):
        board_size = (2, 2)
        players = {
            0: dnbpy.Level1HeuristicPolicy(board_size, random_state=0),
            1: dnbpy.Level2HeuristicPolicy(board_size, random_state=0)
        }
        result = dnbpy.duel(board_size, players)
        self.assertEqual({'tied': 8, 'won': 15, 'lost': 1}, result[1])

        board_size = (3, 3)
        players = {
            0: dnbpy.Level1HeuristicPolicy(board_size, random_state=0),
            1: dnbpy.Level2HeuristicPolicy(board_size, random_state=0)
        }
        result = dnbpy.duel(board_size, players)
        self.assertEqual({'won': 40, 'lost': 8, 'tied': 0}, result[1])

    def test_random_vs_level2(self):
        board_size = (2, 2)
        players = {
            0: dnbpy.RandomPolicy(random_state=0),
            1: dnbpy.Level2HeuristicPolicy(board_size, random_state=0)
        }
        result = dnbpy.duel(board_size, players)
        self.assertEqual({'lost': 3, 'won': 19, 'tied': 2}, result[1])

        board_size = (3, 3)
        players = {
            0: dnbpy.RandomPolicy(random_state=0),
            1: dnbpy.Level2HeuristicPolicy(board_size, random_state=0)
        }
        result = dnbpy.duel(board_size, players)
        self.assertEqual({'lost': 0, 'won': 48, 'tied': 0}, result[1])

    # WARNING: slow test  TODO
    def test_random_vs_level3(self):
        board_size = (3, 3)
        players = {
            0: dnbpy.RandomPolicy(random_state=0),
            1: dnbpy.Level3MinimaxPolicy(board_size, 3, random_state=0)
        }
        result = dnbpy.duel(board_size, players)
        self.assertEqual({'tied': 0, 'lost': 0, 'won': 48}, result[1])

    def test_random_vs_level3update_alpha(self):
        board_size = (3, 3)
        players = {
            0: dnbpy.RandomPolicy(random_state=0),
            1: dnbpy.Level3MinimaxPolicy(board_size, 3, random_state=0, update_alpha=True)
        }
        result = dnbpy.duel(board_size, players)
        self.assertEqual({'tied': 0, 'lost': 0, 'won': 48}, result[1])

    # WARNING: slow test  TODO
    def test_level1_vs_level3(self):
        board_size = (3, 3)
        players = {
            0: dnbpy.Level1HeuristicPolicy(board_size, random_state=0),
            1: dnbpy.Level3MinimaxPolicy(board_size, 3, random_state=0)
        }
        result = dnbpy.duel(board_size, players)
        self.assertEqual({'tied': 0, 'lost': 5, 'won': 43}, result[1])

    def test_level1_vs_level3update_alpha(self):
        board_size = (3, 3)
        players = {
            0: dnbpy.Level1HeuristicPolicy(board_size, random_state=0),
            1: dnbpy.Level3MinimaxPolicy(board_size, 3, random_state=0, update_alpha=True)
        }
        result = dnbpy.duel(board_size, players)
        self.assertEqual({'tied': 0, 'lost': 6, 'won': 42}, result[1])

    # WARNING: slow test  TODO
    def test_level1_vs_level3_variable_depth(self):
        board_size = (3, 3)
        players = {
            0: dnbpy.Level1HeuristicPolicy(board_size, random_state=0),
            1: dnbpy.Level3MinimaxPolicy(board_size, random_state=0)
        }
        result = dnbpy.duel(board_size, players)
        self.assertEqual({'tied': 0, 'lost': 0, 'won': 48}, result[1])

    # WARNING: slow test  TODO
    def test_level2_vs_level3(self):
        board_size = (3, 3)
        players = {
            0: dnbpy.Level2HeuristicPolicy(board_size, random_state=0),
            1: dnbpy.Level3MinimaxPolicy(board_size, 3, random_state=0)
        }
        result = dnbpy.duel(board_size, players)
        self.assertEqual({'tied': 0, 'lost': 10, 'won': 38}, result[1])

    def test_level2_vs_level3update_alpha(self):
        board_size = (3, 3)
        players = {
            0: dnbpy.Level2HeuristicPolicy(board_size, random_state=0),
            1: dnbpy.Level3MinimaxPolicy(board_size, 3, random_state=0, update_alpha=True)
        }
        result = dnbpy.duel(board_size, players)
        self.assertEqual({'tied': 0, 'lost': 13, 'won': 35}, result[1])

    # WARNING: slow test  TODO
    def test_level2_vs_level3_variable_depth(self):
        board_size = (3, 3)
        players = {
            0: dnbpy.Level2HeuristicPolicy(board_size, random_state=0),
            1: dnbpy.Level3MinimaxPolicy(board_size, random_state=0)
        }
        result = dnbpy.duel(board_size, players)
        self.assertEqual({'tied': 0, 'lost': 7, 'won': 41}, result[1])

    # WARNING: slow test  TODO
    def test_level2_vs_level3update_alpha_variable_depth(self):
        board_size = (3, 3)
        players = {
            0: dnbpy.Level2HeuristicPolicy(board_size, random_state=0),
            1: dnbpy.Level3MinimaxPolicy(board_size, random_state=0, update_alpha=True)
        }
        result = dnbpy.duel(board_size, players)
        self.assertEqual({'tied': 0, 'lost': 3, 'won': 45}, result[1])

    # WARNING: slow test  TODO
    def test_level3_vs_level3update_alpha_variable_depth(self):
        board_size = (3, 3)
        players = {
            0: dnbpy.Level3MinimaxPolicy(board_size, random_state=0),
            1: dnbpy.Level3MinimaxPolicy(board_size, random_state=0, update_alpha=True)
        }
        result = dnbpy.duel(board_size, players)
        self.assertEqual({'tied': 0, 'lost': 23, 'won': 25}, result[1])

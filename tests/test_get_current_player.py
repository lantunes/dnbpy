import unittest

import dnbpy


class TestGetCurrentPlayer(unittest.TestCase):

    def test_get_current_player(self):
        game = dnbpy.Game((1, 1), ['player1', 'player2'])
        self.assertEqual(game.get_current_player(), 'player1')
        game.select_edge(0, 'player1')
        self.assertEqual(game.get_current_player(), 'player2')
        game.select_edge(1, 'player2')
        self.assertEqual(game.get_current_player(), 'player1')
        game.select_edge(2, 'player1')
        self.assertEqual(game.get_current_player(), 'player2')
        game.select_edge(3, 'player2')
        self.assertIsNone(game.get_current_player())

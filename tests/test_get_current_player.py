import unittest

import dnbpy


class TestGetCurrentPlayer(unittest.TestCase):

    def test_get_current_player_1x1(self):
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

    def test_get_current_player_2x2(self):
        game = dnbpy.Game((2, 2), ['player1', 'player2'])
        self.assertEqual(game.get_current_player(), 'player1')
        game.select_edge(0, 'player1')
        self.assertEqual(game.get_current_player(), 'player2')
        game.select_edge(2, 'player2')
        self.assertEqual(game.get_current_player(), 'player1')
        game.select_edge(3, 'player1')
        self.assertEqual(game.get_current_player(), 'player2')
        game.select_edge(5, 'player2')
        self.assertEqual(game.get_current_player(), 'player2')

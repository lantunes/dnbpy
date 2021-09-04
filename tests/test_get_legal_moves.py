import unittest

import dnbpy


class TestGetLegalMoves(unittest.TestCase):

    def test_get_legal_moves(self):
        game = dnbpy.Game((1, 1), ['player1', 'player2'])
        self.assertEqual([0, 1, 2, 3], game.get_legal_moves())

        game = dnbpy.Game((1, 1), ['player1', 'player2'], board_state=[0, 1, 1, 0])
        self.assertEqual([0, 3], game.get_legal_moves())

        game.select_edge(0, 'player1')
        self.assertEqual([3], game.get_legal_moves())

        game.select_edge(3, 'player2')
        self.assertEqual([], game.get_legal_moves())

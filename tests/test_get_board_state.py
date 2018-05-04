import unittest

import dnbpy


class TestGetBoardState(unittest.TestCase):

    def test_get_board_state_1x1(self):
        game = dnbpy.Game((1, 1), ['player1', 'player2'])
        board_state = game.get_board_state()
        self.assertEqual(board_state, [0, 0, 0, 0])

    def test_get_board_state_1x2(self):
        game = dnbpy.Game((1, 2), ['player1', 'player2'])
        board_state = game.get_board_state()
        self.assertEqual(board_state, [0, 0, 0, 0, 0, 0, 0])

    def test_get_board_state_2x1(self):
        game = dnbpy.Game((2, 1), ['player1', 'player2'])
        board_state = game.get_board_state()
        self.assertEqual(board_state, [0, 0, 0, 0, 0, 0, 0])

    def test_get_board_state_2x2(self):
        game = dnbpy.Game((2, 2), ['player1', 'player2'])
        board_state = game.get_board_state()
        self.assertEqual(board_state, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_get_board_state_2x3(self):
        game = dnbpy.Game((2, 3), ['player1', 'player2'])
        board_state = game.get_board_state()
        self.assertEqual(board_state, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_get_board_state_3x2(self):
        game = dnbpy.Game((3, 2), ['player1', 'player2'])
        board_state = game.get_board_state()
        self.assertEqual(board_state, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_get_board_state_3x3(self):
        game = dnbpy.Game((3, 3), ['player1', 'player2'])
        board_state = game.get_board_state()
        self.assertEqual(board_state, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

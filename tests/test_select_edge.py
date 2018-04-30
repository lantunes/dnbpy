import unittest

import dnbpy


class TestSelectEdge(unittest.TestCase):

    def test_select_edge_2x2(self):
        game = dnbpy.Game((2, 2), ['player1', 'player2'])
        next_player, boxes_made = game.select_edge(2, 'player1')
        self.assertEqual(next_player, 'player2')
        self.assertEqual(boxes_made, 0)
        board_state = game.get_board_state()
        self.assertEqual(board_state, [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        next_player, boxes_made = game.select_edge(5, 'player2')
        self.assertEqual(next_player, 'player1')
        self.assertEqual(boxes_made, 0)
        board_state = game.get_board_state()
        self.assertEqual(board_state, [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0])

    def test_select_edge_player_not_recognized(self):
        with self.assertRaises(Exception) as e:
            game = dnbpy.Game((2, 2), ['player1', 'player2'])
            game.select_edge(2, 'player3')
        self.assertTrue(("next player to play is: player1" in str(e.exception)))

    def test_select_non_existent_edge(self):
        with self.assertRaises(Exception) as e:
            game = dnbpy.Game((2, 2), ['player1', 'player2'])
            game.select_edge(12, 'player1')
        self.assertTrue(("invalid edge index: 12" in str(e.exception)))

    def test_select_taken_edge(self):
        with self.assertRaises(Exception) as e:
            game = dnbpy.Game((2, 2), ['player1', 'player2'])
            game.select_edge(2, 'player1')
            game.select_edge(2, 'player2')
        self.assertTrue(("edge already selected: 2" in str(e.exception)))

    def test_select_edge_returns_next_player(self):
        game = dnbpy.Game((1, 1), ['player1', 'player2'])
        next_player, boxes_made = game.select_edge(0, 'player1')
        self.assertEqual(next_player, 'player2')
        self.assertEqual(boxes_made, 0)
        next_player, boxes_made = game.select_edge(1, 'player2')
        self.assertEqual(next_player, 'player1')
        self.assertEqual(boxes_made, 0)
        next_player, boxes_made = game.select_edge(2, 'player1')
        self.assertEqual(next_player, 'player2')
        self.assertEqual(boxes_made, 0)
        next_player, boxes_made = game.select_edge(3, 'player2')
        self.assertEqual(boxes_made, 1)
        self.assertIsNone(next_player)
        self.assertEqual(game.get_score('player1'), 0)
        self.assertEqual(game.get_score('player2'), 1)
        with self.assertRaises(Exception) as e:
            game.select_edge(1, 'player1')
        self.assertTrue(("game is finished" in str(e.exception)))

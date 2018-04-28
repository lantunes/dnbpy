import unittest

import dnbpy


class TestGetScore(unittest.TestCase):

    def test_get_score_2x2(self):
        game = dnbpy.Game((2, 2), ['player1', 'player2'])
        game.select_edge(0, 'player1')
        game.select_edge(2, 'player2')
        game.select_edge(3, 'player1')
        game.select_edge(5, 'player2')
        score = game.get_score('player1')
        self.assertEqual(score, 0)
        score = game.get_score('player2')
        self.assertEqual(score, 1)
        self.assertFalse(game.is_finished())
        game.select_edge(11, 'player2')
        game.select_edge(6, 'player1')
        game.select_edge(10, 'player2')
        game.select_edge(8, 'player1')
        score = game.get_score('player1')
        self.assertEqual(score, 0)
        score = game.get_score('player2')
        self.assertEqual(score, 1)
        game.select_edge(9, 'player2')
        game.select_edge(7, 'player2')
        score = game.get_score('player1')
        self.assertEqual(score, 0)
        score = game.get_score('player2')
        self.assertEqual(score, 3)
        game.select_edge(4, 'player2')
        game.select_edge(1, 'player1')
        score = game.get_score('player1')
        self.assertEqual(score, 1)
        score = game.get_score('player2')
        self.assertEqual(score, 3)
        self.assertTrue(game.is_finished())
        with self.assertRaises(Exception) as e:
            game.select_edge(1, 'player1')
        self.assertTrue(("game is finished" in str(e.exception)))
        boxes = game.get_boxes('player1')
        self.assertEqual(len(boxes), 1)
        self.assertEqual(str(boxes[0]), '1-3-4-6')
        boxes = game.get_boxes('player2')
        self.assertEqual(len(boxes), 3)
        self.assertEqual(str(boxes[0]), '0-2-3-5')
        self.assertEqual(str(boxes[1]), '6-8-9-11')
        self.assertEqual(str(boxes[2]), '5-7-8-10')

    def test_get_score_player_not_recognized(self):
        with self.assertRaises(Exception) as e:
            game = dnbpy.Game((2, 2), ['player1', 'player2'])
            game.get_score('player3')
        self.assertTrue(("player not recognized: player3" in str(e.exception)))

import unittest

import dnbpy


class TestGetBoxes(unittest.TestCase):

    def test_get_boxes_2x2(self):
        game = dnbpy.Game((2, 2), ['player1', 'player2'])
        game.select_edge(0, 'player1')
        game.select_edge(2, 'player2')
        game.select_edge(3, 'player1')
        game.select_edge(5, 'player2')
        boxes = game.get_boxes('player1')
        self.assertEqual(len(boxes), 0)
        boxes = game.get_boxes('player2')
        self.assertEqual(len(boxes), 1)
        self.assertEqual(str(boxes[0]), '0-2-3-5')

    def test_get_boxes_player_not_recognized(self):
        with self.assertRaises(Exception) as e:
            game = dnbpy.Game((2, 2), ['player1', 'player2'])
            game.get_boxes('player3')
        self.assertTrue(("player not recognized: player3" in str(e.exception)))

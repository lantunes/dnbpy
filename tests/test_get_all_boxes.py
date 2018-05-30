import unittest

import dnbpy


class TestGetAllBoxes(unittest.TestCase):

    def test_get_all_boxes_1x1(self):
        game = dnbpy.Game((1, 1), ['player1', 'player2'])
        boxes = game.get_all_boxes()
        self.assertTrue(len(boxes) == 1)
        self.assertEqual(str(boxes[0]), '0-1-2-3')

    def test_get_all_boxes_2x2(self):
        game = dnbpy.Game((2, 2), ['player1', 'player2'])
        boxes = game.get_all_boxes()
        self.assertTrue(len(boxes) == 4)
        self.assertEqual(str(boxes[0]), '0-2-3-5')
        self.assertEqual(str(boxes[1]), '1-3-4-6')
        self.assertEqual(str(boxes[2]), '5-7-8-10')
        self.assertEqual(str(boxes[3]), '6-8-9-11')

    def test_get_all_boxes_2x3(self):
        game = dnbpy.Game((2, 3), ['player1', 'player2'])
        boxes = game.get_all_boxes()
        self.assertTrue(len(boxes) == 6)
        self.assertEqual(str(boxes[0]), '0-3-4-7')
        self.assertEqual(str(boxes[1]), '1-4-5-8')
        self.assertEqual(str(boxes[2]), '2-5-6-9')
        self.assertEqual(str(boxes[3]), '7-10-11-14')
        self.assertEqual(str(boxes[4]), '8-11-12-15')
        self.assertEqual(str(boxes[5]), '9-12-13-16')

    def test_get_all_boxes_3x3(self):
        game = dnbpy.Game((3, 3), ['player1', 'player2'])
        boxes = game.get_all_boxes()
        self.assertTrue(len(boxes) == 9)
        self.assertEqual(str(boxes[0]), '0-3-4-7')
        self.assertEqual(str(boxes[1]), '1-4-5-8')
        self.assertEqual(str(boxes[2]), '2-5-6-9')
        self.assertEqual(str(boxes[3]), '7-10-11-14')
        self.assertEqual(str(boxes[4]), '8-11-12-15')
        self.assertEqual(str(boxes[5]), '9-12-13-16')
        self.assertEqual(str(boxes[6]), '14-17-18-21')
        self.assertEqual(str(boxes[7]), '15-18-19-22')
        self.assertEqual(str(boxes[8]), '16-19-20-23')

import unittest

import dnbpy


class TestToString(unittest.TestCase):

    def test_to_string_3x3(self):
        game = dnbpy.Game((3, 3), ['p1', 'p2'])
        game.select_edge(0, 'p1')
        game.select_edge(9, 'p2')
        game.select_edge(10, 'p1')
        game.select_edge(13, 'p2')
        game.select_edge(2, 'p1')
        game.select_edge(6, 'p2')
        game.select_edge(5, 'p1')
        game.select_edge(12, 'p1')
        game.select_edge(16, 'p2')
        expected = (
            "*    -    *    1    *    -    *  \n"
            "                                 \n"
            "3         4         |   p1    |  \n"
            "                                 \n"
            "*    7    *    8    *    -    *  \n"
            "                                 \n"
            "|         11        |   p2    |  \n"
            "                                 \n"
            "*    14   *    15   *    -    *  \n"
            "                                 \n"
            "17        18        19        20 \n"
            "                                 \n"
            "*    21   *    22   *    23   *  \n"
            "                                 \n"
            "p1: 1, p2: 1"
        )
        self.assertEqual(str(game), expected)

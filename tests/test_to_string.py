import unittest

import dnbpy


class TestToString(unittest.TestCase):

    def test_to_string_1x1(self):
        game = dnbpy.Game((1, 1), ['p1', 'p2'])
        game.select_edge(0, 'p1')
        game.select_edge(1, 'p2')
        expected = (
            "*---------*  \n"
            "|            \n"
            "|         2  \n"
            "|            \n"
            "*    3    *  \n"
            "\n"
            "p1: 0, p2: 0"
        )
        self.assertEqual(str(game), expected)

    def test_to_string_1x2(self):
        game = dnbpy.Game((1, 2), ['p1', 'p2'])
        game.select_edge(1, 'p1')
        game.select_edge(3, 'p2')
        game.select_edge(6, 'p1')
        expected = (
            "*    0    *---------*  \n"
            "          |            \n"
            "2         |         4  \n"
            "          |            \n"
            "*    5    *---------*  \n"
            "\n"
            "p1: 0, p2: 0"
        )
        self.assertEqual(str(game), expected)

    def test_to_string_2x1(self):
        game = dnbpy.Game((2, 1), ['p1', 'p2'])
        game.select_edge(1, 'p1')
        game.select_edge(3, 'p2')
        game.select_edge(6, 'p1')
        expected = (
            "*    0    *  \n"
            "|            \n"
            "|         2  \n"
            "|            \n"
            "*---------*  \n"
            "             \n"
            "4         5  \n"
            "             \n"
            "*---------*  \n"
            "\n"
            "p1: 0, p2: 0"
        )
        self.assertEqual(str(game), expected)

    def test_to_string_2x2(self):
        game = dnbpy.Game((2, 2), ['p1', 'p2'])
        game.select_edge(1, 'p1')
        game.select_edge(3, 'p2')
        game.select_edge(4, 'p1')
        game.select_edge(6, 'p2')
        game.select_edge(11, 'p2')
        expected = (
            "*    0    *---------*  \n"
            "          |         |  \n"
            "2         |   p2    |  \n"
            "          |         |  \n"
            "*    5    *---------*  \n"
            "                       \n"
            "7         8         9  \n"
            "                       \n"
            "*    10   *---------*  \n"
            "\n"
            "p1: 0, p2: 1"
        )
        self.assertEqual(str(game), expected)

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
            "*---------*    1    *---------*  \n"
            "                    |         |  \n"
            "3         4         |   p1    |  \n"
            "                    |         |  \n"
            "*    7    *    8    *---------*  \n"
            "|                   |         |  \n"
            "|         11        |   p2    |  \n"
            "|                   |         |  \n"
            "*    14   *    15   *---------*  \n"
            "                                 \n"
            "17        18        19        20 \n"
            "                                 \n"
            "*    21   *    22   *    23   *  \n"
            "\n"
            "p1: 1, p2: 1"
        )
        self.assertEqual(str(game), expected)

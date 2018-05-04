import unittest

import dnbpy


class TestCreateEngine(unittest.TestCase):

    def test_create_engine_checks_args(self):
        with self.assertRaises(Exception):
            dnbpy.Game(1, ['player1', 'player2'])
        with self.assertRaises(Exception):
            dnbpy.Game((1, 2, 3), ['player1', 'player2'])
        with self.assertRaises(Exception):
            dnbpy.Game((0, 1), ['player1', 'player2'])
        with self.assertRaises(Exception):
            dnbpy.Game((1, 0), ['player1', 'player2'])
        with self.assertRaises(Exception):
            dnbpy.Game((-1, 1), ['player1', 'player2'])
        with self.assertRaises(Exception):
            dnbpy.Game((1, -1), ['player1', 'player2'])
        with self.assertRaises(Exception):
            dnbpy.Game((1, 1.2), ['player1', 'player2'])
        with self.assertRaises(Exception):
            dnbpy.Game((1, 1), [])
        with self.assertRaises(Exception):
            dnbpy.Game((1, 1), ['player1'])
        with self.assertRaises(Exception):
            dnbpy.Game((1, 1), None)
        with self.assertRaises(Exception):
            dnbpy.Game(None, ['player1', 'player2'])

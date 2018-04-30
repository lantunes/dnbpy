import unittest

import dnbpy


class TestBoardStateToString(unittest.TestCase):

    def test_as_string(self):
        self.assertEqual(dnbpy.as_string([0, 1, 0, 1]), '0101')
        self.assertEqual(dnbpy.as_string([0, 1, 0, 1, 0, 0]), '010100')

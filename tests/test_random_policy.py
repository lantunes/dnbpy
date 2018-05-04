import unittest

import dnbpy


class TestRandomPolicy(unittest.TestCase):

    def test_random_policy(self):
        policy = dnbpy.RandomPolicy()
        board_state = [0, 1, 1, 0]
        edge = policy.select_edge(board_state)
        self.assertTrue(edge == 0 or edge == 3)

    def test_random_policy_raises_exception_when_state_complete(self):
        policy = dnbpy.RandomPolicy()
        board_state = [1, 1, 1, 1]
        with self.assertRaises(Exception) as e:
            policy.select_edge(board_state)
        self.assertTrue(("there are no edges to select" in str(e.exception)))

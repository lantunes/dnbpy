import unittest

import dnbpy


class TestLevel1HeuristicPolicy(unittest.TestCase):

    def test_select_edge(self):
        policy = dnbpy.Level1HeuristicPolicy(board_size=(3, 3))

        """
        *---------*---------*---------*  
        |         |                      
        |   $L1   |         5         6  
        |         |                      
        *---------*---------*    9    *  
                                      |  
        10        11        12        |  
                                      |  
        *---------*    15   *    16   *  
                                      |  
        17        18        19        |  
                                      |  
        *    21   *    22   *---------*  
        """
        board_state = [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1]

        edge = policy.select_edge(board_state)

        self.assertEqual(5, edge)

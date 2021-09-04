import unittest

import dnbpy


class TestLevel2HeuristicPolicy(unittest.TestCase):

    def test_select_edge(self):
        policy = dnbpy.Level2HeuristicPolicy(board_size=(3, 3))

        """
        *---------*---------*---------*  
        |         |                      
        |   $L2   |         5         6  
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

        """
        *---------*---------*---------*  
        |         |         |             
        |   $L2   |   $L2   |         6  
        |         |         |            
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
        board_state = [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1]

        edge = policy.select_edge(board_state)

        self.assertTrue(edge not in [6, 9, 10, 11, 16, 19])

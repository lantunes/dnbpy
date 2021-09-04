import unittest

import dnbpy


class TestStringsAndCoins(unittest.TestCase):

    def test_get_strings_and_coins_1x1(self):
        game = dnbpy.Game((1, 1), ['p1', 'p2'])
        """
        *    0    *  
             
        1         2  
                     
        *    3    *  
        """
        strings_and_coins = dnbpy.StringsAndCoins(game)
        self.assertEqual(strings_and_coins.nodes, [
            '0-1-2-3',
            'ground-1',
            'ground-2',
            'ground-0',
            'ground-3',
        ])
        self.assertEqual(strings_and_coins.edges, [
            ('ground-1', '0-1-2-3'),
            ('ground-2', '0-1-2-3'),
            ('ground-0', '0-1-2-3'),
            ('ground-3', '0-1-2-3')
        ])
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-0'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-1'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-2'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-3'))
        self.assertEqual(4, strings_and_coins.get_node_degree('0-1-2-3'))
        self.assertEqual(strings_and_coins.to_adjacency_matrix(), [
            [0, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0]
        ])

    def test_get_strings_and_coins_1x1_edges_selected(self):
        game = dnbpy.Game((1, 1), ['p1', 'p2'])
        game.select_edge(0, 'p1')
        game.select_edge(2, 'p2')
        """
        *---------*  
                  |  
        1         |  
                  |  
        *    3    *
        """
        strings_and_coins = dnbpy.StringsAndCoins(game)
        self.assertEqual(strings_and_coins.nodes, [
            '0-1-2-3',
            'ground-1',
            'ground-2',
            'ground-0',
            'ground-3',
        ])
        self.assertEqual(strings_and_coins.edges, [
            ('ground-1', '0-1-2-3'),
            ('ground-3', '0-1-2-3')
        ])
        self.assertEqual(0, strings_and_coins.get_node_degree('ground-0'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-1'))
        self.assertEqual(0, strings_and_coins.get_node_degree('ground-2'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-3'))
        self.assertEqual(2, strings_and_coins.get_node_degree('0-1-2-3'))
        self.assertEqual(strings_and_coins.to_adjacency_matrix(), [
            [0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0]
        ])

    def test_get_strings_and_coins_2x3(self):
        game = dnbpy.Game((2, 3), ['p1', 'p2'])
        """
        *    0    *    1    *    2    *  
                                 
        3         4         5         6  
                                         
        *    7    *    8    *    9    *  
                                         
        10        11        12        13 
                                         
        *    14   *    15   *    16   *  
        """
        strings_and_coins = dnbpy.StringsAndCoins(game)
        self.assertEqual(strings_and_coins.nodes, [
            '0-3-4-7',
            'ground-3',
            'ground-0',
            '1-4-5-8',
            'ground-1',
            '2-5-6-9',
            'ground-6',
            'ground-2',
            '7-10-11-14',
            'ground-10',
            'ground-14',
            '8-11-12-15',
            'ground-15',
            '9-12-13-16',
            'ground-13',
            'ground-16'
        ])
        self.assertEqual(strings_and_coins.edges, [
            ('ground-3', '0-3-4-7'),
            ('ground-0', '0-3-4-7'),
            ('0-3-4-7', '1-4-5-8'),
            ('ground-1', '1-4-5-8'),
            ('1-4-5-8', '2-5-6-9'),
            ('ground-6', '2-5-6-9'),
            ('ground-2', '2-5-6-9'),
            ('0-3-4-7', '7-10-11-14'),
            ('ground-10', '7-10-11-14'),
            ('ground-14', '7-10-11-14'),
            ('1-4-5-8', '8-11-12-15'),
            ('7-10-11-14', '8-11-12-15'),
            ('ground-15', '8-11-12-15'),
            ('2-5-6-9', '9-12-13-16'),
            ('8-11-12-15', '9-12-13-16'),
            ('ground-13', '9-12-13-16'),
            ('ground-16', '9-12-13-16')
        ])
        self.assertEqual(4, strings_and_coins.get_node_degree('0-3-4-7'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-3'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-0'))
        self.assertEqual(4, strings_and_coins.get_node_degree('1-4-5-8'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-1'))
        self.assertEqual(4, strings_and_coins.get_node_degree('2-5-6-9'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-6'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-2'))
        self.assertEqual(4, strings_and_coins.get_node_degree('7-10-11-14'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-10'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-14'))
        self.assertEqual(4, strings_and_coins.get_node_degree('8-11-12-15'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-15'))
        self.assertEqual(4, strings_and_coins.get_node_degree('9-12-13-16'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-13'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-16'))
        self.assertEqual(strings_and_coins.to_adjacency_matrix(), [
            [0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        ])

    def test_get_strings_and_coins_2x3_edges_selected(self):
        game = dnbpy.Game((2, 3), ['p1', 'p2'])
        game.select_edge(4, 'p1')
        game.select_edge(12, 'p2')
        game.select_edge(9, 'p1')
        game.select_edge(1, 'p2')
        """
        *    0    *---------*    2    *  
                  |                      
        3         |         5         6  
                  |                      
        *    7    *    8    *---------*  
                            |            
        10        11        |         13 
                            |            
        *    14   *    15   *    16   *  
        """
        strings_and_coins = dnbpy.StringsAndCoins(game)
        self.assertEqual(strings_and_coins.nodes, [
            '0-3-4-7',
            'ground-3',
            'ground-0',
            '1-4-5-8',
            'ground-1',
            '2-5-6-9',
            'ground-6',
            'ground-2',
            '7-10-11-14',
            'ground-10',
            'ground-14',
            '8-11-12-15',
            'ground-15',
            '9-12-13-16',
            'ground-13',
            'ground-16'
        ])
        self.assertEqual(strings_and_coins.edges, [
            ('ground-3', '0-3-4-7'),
            ('ground-0', '0-3-4-7'),
            ('1-4-5-8', '2-5-6-9'),
            ('ground-6', '2-5-6-9'),
            ('ground-2', '2-5-6-9'),
            ('0-3-4-7', '7-10-11-14'),
            ('ground-10', '7-10-11-14'),
            ('ground-14', '7-10-11-14'),
            ('1-4-5-8', '8-11-12-15'),
            ('7-10-11-14', '8-11-12-15'),
            ('ground-15', '8-11-12-15'),
            ('ground-13', '9-12-13-16'),
            ('ground-16', '9-12-13-16')
        ])
        self.assertEqual(3, strings_and_coins.get_node_degree('0-3-4-7'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-3'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-0'))
        self.assertEqual(2, strings_and_coins.get_node_degree('1-4-5-8'))
        self.assertEqual(0, strings_and_coins.get_node_degree('ground-1'))
        self.assertEqual(3, strings_and_coins.get_node_degree('2-5-6-9'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-6'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-2'))
        self.assertEqual(4, strings_and_coins.get_node_degree('7-10-11-14'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-10'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-14'))
        self.assertEqual(3, strings_and_coins.get_node_degree('8-11-12-15'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-15'))
        self.assertEqual(2, strings_and_coins.get_node_degree('9-12-13-16'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-13'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-16'))
        self.assertEqual(strings_and_coins.to_adjacency_matrix(), [
            [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        ])

    def test_get_strings_and_coins_3x2(self):
        game = dnbpy.Game((3, 2), ['p1', 'p2'])
        """
        *    0    *    1    *  
                               
        2         3         4  
                               
        *    5    *    6    *  
                               
        7         8         9  
                               
        *    10   *    11   *  
                               
        12        13        14 
                               
        *    15   *    16   * 
        """
        strings_and_coins = dnbpy.StringsAndCoins(game)
        self.assertEqual(strings_and_coins.nodes, [
            '0-2-3-5',
            'ground-2',
            'ground-0',
            '1-3-4-6',
            'ground-4',
            'ground-1',
            '5-7-8-10',
            'ground-7',
            '6-8-9-11',
            'ground-9',
            '10-12-13-15',
            'ground-12',
            'ground-15',
            '11-13-14-16',
            'ground-14',
            'ground-16'
        ])
        self.assertEqual(strings_and_coins.edges, [
            ('ground-2', '0-2-3-5'),
            ('ground-0', '0-2-3-5'),
            ('0-2-3-5', '1-3-4-6'),
            ('ground-4', '1-3-4-6'),
            ('ground-1', '1-3-4-6'),
            ('0-2-3-5', '5-7-8-10'),
            ('ground-7', '5-7-8-10'),
            ('1-3-4-6', '6-8-9-11'),
            ('5-7-8-10', '6-8-9-11'),
            ('ground-9', '6-8-9-11'),
            ('5-7-8-10', '10-12-13-15'),
            ('ground-12', '10-12-13-15'),
            ('ground-15', '10-12-13-15'),
            ('6-8-9-11', '11-13-14-16'),
            ('10-12-13-15', '11-13-14-16'),
            ('ground-14', '11-13-14-16'),
            ('ground-16', '11-13-14-16')
        ])
        self.assertEqual(4, strings_and_coins.get_node_degree('0-2-3-5'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-2'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-0'))
        self.assertEqual(4, strings_and_coins.get_node_degree('1-3-4-6'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-4'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-1'))
        self.assertEqual(4, strings_and_coins.get_node_degree('5-7-8-10'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-7'))
        self.assertEqual(4, strings_and_coins.get_node_degree('6-8-9-11'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-9'))
        self.assertEqual(4, strings_and_coins.get_node_degree('10-12-13-15'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-12'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-15'))
        self.assertEqual(4, strings_and_coins.get_node_degree('11-13-14-16'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-14'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-16'))
        self.assertEqual(strings_and_coins.to_adjacency_matrix(), [
            [0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        ])

    def test_get_strings_and_coins_3x2_edges_selected(self):
        game = dnbpy.Game((3, 2), ['p1', 'p2'])
        game.select_edge(0, 'p1')
        game.select_edge(4, 'p2')
        game.select_edge(7, 'p1')
        game.select_edge(13, 'p2')
        game.select_edge(16, 'p1')
        """
        *---------*    1    *  
                            |  
        2         3         |  
                            |  
        *    5    *    6    *  
        |                      
        |         8         9  
        |                      
        *    10   *    11   *  
                  |            
        12        |         14 
                  |            
        *    15   *---------* 
        """
        strings_and_coins = dnbpy.StringsAndCoins(game)
        self.assertEqual(strings_and_coins.nodes, [
            '0-2-3-5',
            'ground-2',
            'ground-0',
            '1-3-4-6',
            'ground-4',
            'ground-1',
            '5-7-8-10',
            'ground-7',
            '6-8-9-11',
            'ground-9',
            '10-12-13-15',
            'ground-12',
            'ground-15',
            '11-13-14-16',
            'ground-14',
            'ground-16'
        ])
        self.assertEqual(strings_and_coins.edges, [
            ('ground-2', '0-2-3-5'),
            ('0-2-3-5', '1-3-4-6'),
            ('ground-1', '1-3-4-6'),
            ('0-2-3-5', '5-7-8-10'),
            ('1-3-4-6', '6-8-9-11'),
            ('5-7-8-10', '6-8-9-11'),
            ('ground-9', '6-8-9-11'),
            ('5-7-8-10', '10-12-13-15'),
            ('ground-12', '10-12-13-15'),
            ('ground-15', '10-12-13-15'),
            ('6-8-9-11', '11-13-14-16'),
            ('ground-14', '11-13-14-16')
        ])
        self.assertEqual(3, strings_and_coins.get_node_degree('0-2-3-5'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-2'))
        self.assertEqual(0, strings_and_coins.get_node_degree('ground-0'))
        self.assertEqual(3, strings_and_coins.get_node_degree('1-3-4-6'))
        self.assertEqual(0, strings_and_coins.get_node_degree('ground-4'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-1'))
        self.assertEqual(3, strings_and_coins.get_node_degree('5-7-8-10'))
        self.assertEqual(0, strings_and_coins.get_node_degree('ground-7'))
        self.assertEqual(4, strings_and_coins.get_node_degree('6-8-9-11'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-9'))
        self.assertEqual(3, strings_and_coins.get_node_degree('10-12-13-15'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-12'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-15'))
        self.assertEqual(2, strings_and_coins.get_node_degree('11-13-14-16'))
        self.assertEqual(1, strings_and_coins.get_node_degree('ground-14'))
        self.assertEqual(0, strings_and_coins.get_node_degree('ground-16'))
        self.assertEqual(strings_and_coins.to_adjacency_matrix(), [
            [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])

    def test_get_long_chains_2x3(self):
        game = dnbpy.Game((2, 3), ['p1', 'p2'])
        game.select_edge(4, 'p1')
        game.select_edge(12, 'p2')
        game.select_edge(9, 'p1')
        game.select_edge(1, 'p2')
        game.select_edge(11, 'p1')
        game.select_edge(3, 'p2')
        game.select_edge(10, 'p1')
        game.select_edge(2, 'p2')
        """
        This board contains a single chain (i.e. a string of at least 3 non-ground nodes):
        *    0    *---------*---------*  
        |         |                      
        |         |         5         6  
        |         |                      
        *    7    *    8    *---------*  
        |         |         |            
        |         |         |         13 
        |         |         |            
        *    14   *    15   *    16   *  
        """
        strings_and_coins = dnbpy.StringsAndCoins(game)
        chains = strings_and_coins.get_long_chains()
        self.assertEqual([['2-5-6-9', '1-4-5-8', '8-11-12-15']], chains)

    def test_get_long_chains_3x3(self):
        game = dnbpy.Game((3, 3), ['p1', 'p2'])
        game.select_edge(4, 'p1')
        game.select_edge(12, 'p2')
        game.select_edge(9, 'p1')
        game.select_edge(1, 'p2')
        game.select_edge(11, 'p1')
        game.select_edge(3, 'p2')
        game.select_edge(10, 'p1')
        game.select_edge(2, 'p2')
        """
        This board contains a single chain (i.e. a string of at least 3 non-ground nodes):
        *    0    *---------*---------*  
        |         |                      
        |         |         5         6  
        |         |                      
        *    7    *    8    *---------*  
        |         |         |            
        |         |         |         13 
        |         |         |            
        *    14   *    15   *    16   *  
                                         
        17        18        19        20 
                                         
        *    21   *    22   *    23   *  
        """
        strings_and_coins = dnbpy.StringsAndCoins(game)
        chains = strings_and_coins.get_long_chains()
        self.assertEqual([['2-5-6-9', '1-4-5-8', '8-11-12-15']], chains)

    def test_get_long_chains_3x3_multiple(self):
        game = dnbpy.Game((3, 3), ['p1', 'p2'])
        game.select_edge(3, 'p1')
        game.select_edge(10, 'p2')
        game.select_edge(17, 'p1')
        game.select_edge(4, 'p2')
        game.select_edge(11, 'p1')
        game.select_edge(18, 'p2')
        game.select_edge(5, 'p1')
        game.select_edge(12, 'p2')
        game.select_edge(19, 'p1')
        game.select_edge(6, 'p2')
        game.select_edge(13, 'p1')
        game.select_edge(20, 'p2')
        """
        This board contains 3 chains (i.e. a string of at least 3 non-ground nodes):
        *    0    *    1    *    2    *  
        |         |         |         |  
        |         |         |         |  
        |         |         |         |  
        *    7    *    8    *    9    *  
        |         |         |         |  
        |         |         |         |  
        |         |         |         |  
        *    14   *    15   *    16   *  
        |         |         |         |  
        |         |         |         |  
        |         |         |         |  
        *    21   *    22   *    23   *      
        """
        strings_and_coins = dnbpy.StringsAndCoins(game)
        chains = strings_and_coins.get_long_chains()
        self.assertEqual([
            ['0-3-4-7', '7-10-11-14', '14-17-18-21'],
            ['1-4-5-8', '8-11-12-15', '15-18-19-22'],
            ['2-5-6-9', '9-12-13-16', '16-19-20-23']
        ], chains)

    def test_get_long_chains_2x3_closed(self):
        game = dnbpy.Game((2, 3), ['p1', 'p2'])
        game.select_edge(7, 'p1')
        game.select_edge(8, 'p2')
        game.select_edge(9, 'p1')
        game.select_edge(10, 'p2')
        game.select_edge(13, 'p1')
        game.select_edge(14, 'p2')
        game.select_edge(15, 'p1')
        """
        This board contains a single chain (i.e. a string of at least 3 non-ground nodes):
        *    0    *    1    *    2    *  
                                     
        3         4         5         6  
                                     
        *---------*---------*---------*  
        |                             |  
        |         11        12        |  
        |                             |  
        *---------*---------*    16   *  
        """
        strings_and_coins = dnbpy.StringsAndCoins(game)
        chains = strings_and_coins.get_long_chains()
        self.assertEqual([['7-10-11-14', '8-11-12-15', '9-12-13-16']], chains)
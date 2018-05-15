import unittest

import dnbpy


class TestDNBPy(unittest.TestCase):

    def test_create_engine_checks_args(self):
        with self.assertRaises(Exception):
            dnbpy.GameEngine(1, ['player1', 'player2'])
        with self.assertRaises(Exception):
            dnbpy.GameEngine((1, 2, 3), ['player1', 'player2'])
        with self.assertRaises(Exception):
            dnbpy.GameEngine((0, 1), ['player1', 'player2'])
        with self.assertRaises(Exception):
            dnbpy.GameEngine((1, 0), ['player1', 'player2'])
        with self.assertRaises(Exception):
            dnbpy.GameEngine((-1, 1), ['player1', 'player2'])
        with self.assertRaises(Exception):
            dnbpy.GameEngine((1, -1), ['player1', 'player2'])
        with self.assertRaises(Exception):
            dnbpy.GameEngine((1, 1.2), ['player1', 'player2'])
        with self.assertRaises(Exception):
            dnbpy.GameEngine((1, 1), [])
        with self.assertRaises(Exception):
            dnbpy.GameEngine((1, 1), ['player1'])
        with self.assertRaises(Exception):
            dnbpy.GameEngine((1, 1), None)
        with self.assertRaises(Exception):
            dnbpy.GameEngine(None, ['player1', 'player2'])

    def test_get_board_state_1x1(self):
        game_engine = dnbpy.GameEngine((1, 1), ['player1', 'player2'])
        board_state = game_engine.get_board_state()
        self.assertEqual(board_state, [0, 0, 0, 0])

    def test_get_board_state_1x2(self):
        game_engine = dnbpy.GameEngine((1, 2), ['player1', 'player2'])
        board_state = game_engine.get_board_state()
        self.assertEqual(board_state, [0, 0, 0, 0, 0, 0, 0])

    def test_get_board_state_2x1(self):
        game_engine = dnbpy.GameEngine((2, 1), ['player1', 'player2'])
        board_state = game_engine.get_board_state()
        self.assertEqual(board_state, [0, 0, 0, 0, 0, 0, 0])

    def test_get_board_state_2x2(self):
        game_engine = dnbpy.GameEngine((2, 2), ['player1', 'player2'])
        board_state = game_engine.get_board_state()
        self.assertEqual(board_state, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_get_board_state_2x3(self):
        game_engine = dnbpy.GameEngine((2, 3), ['player1', 'player2'])
        board_state = game_engine.get_board_state()
        self.assertEqual(board_state, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_get_board_state_3x2(self):
        game_engine = dnbpy.GameEngine((3, 2), ['player1', 'player2'])
        board_state = game_engine.get_board_state()
        self.assertEqual(board_state, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_get_board_state_3x3(self):
        game_engine = dnbpy.GameEngine((3, 3), ['player1', 'player2'])
        board_state = game_engine.get_board_state()
        self.assertEqual(board_state, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_select_edge_2x2(self):
        game_engine = dnbpy.GameEngine((2, 2), ['player1', 'player2'])
        next_player = game_engine.select_edge(2, 'player1')
        self.assertEqual(next_player, 'player2')
        board_state = game_engine.get_board_state()
        self.assertEqual(board_state, [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        next_player = game_engine.select_edge(5, 'player2')
        self.assertEqual(next_player, 'player1')
        board_state = game_engine.get_board_state()
        self.assertEqual(board_state, [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0])

    def test_select_edge_player_not_recognized(self):
        with self.assertRaises(Exception) as e:
            game_engine = dnbpy.GameEngine((2, 2), ['player1', 'player2'])
            game_engine.select_edge(2, 'player3')
        self.assertTrue(("next player to play is: player1" in str(e.exception)))

    def test_select_non_existent_edge(self):
        with self.assertRaises(Exception) as e:
            game_engine = dnbpy.GameEngine((2, 2), ['player1', 'player2'])
            game_engine.select_edge(12, 'player1')
        self.assertTrue(("invalid edge index: 12" in str(e.exception)))

    def test_select_taken_edge(self):
        with self.assertRaises(Exception) as e:
            game_engine = dnbpy.GameEngine((2, 2), ['player1', 'player2'])
            game_engine.select_edge(2, 'player1')
            game_engine.select_edge(2, 'player2')
        self.assertTrue(("edge already selected: 2" in str(e.exception)))

    def test_get_all_boxes_1x1(self):
        game_engine = dnbpy.GameEngine((1, 1), ['player1', 'player2'])
        boxes = game_engine.get_all_boxes()
        self.assertTrue(len(boxes) == 1)
        self.assertEqual(str(boxes[0]), '0-1-2-3')

    def test_get_all_boxes_2x2(self):
        game_engine = dnbpy.GameEngine((2, 2), ['player1', 'player2'])
        boxes = game_engine.get_all_boxes()
        self.assertTrue(len(boxes) == 4)
        self.assertEqual(str(boxes[0]), '0-2-3-5')
        self.assertEqual(str(boxes[1]), '1-3-4-6')
        self.assertEqual(str(boxes[2]), '5-7-8-10')
        self.assertEqual(str(boxes[3]), '6-8-9-11')

    def test_get_all_boxes_2x3(self):
        game_engine = dnbpy.GameEngine((2, 3), ['player1', 'player2'])
        boxes = game_engine.get_all_boxes()
        self.assertTrue(len(boxes) == 6)
        self.assertEqual(str(boxes[0]), '0-3-4-7')
        self.assertEqual(str(boxes[1]), '1-4-5-8')
        self.assertEqual(str(boxes[2]), '2-5-6-9')
        self.assertEqual(str(boxes[3]), '7-10-11-14')
        self.assertEqual(str(boxes[4]), '8-11-12-15')
        self.assertEqual(str(boxes[5]), '9-12-13-16')

    def test_get_all_boxes_3x3(self):
        game_engine = dnbpy.GameEngine((3, 3), ['player1', 'player2'])
        boxes = game_engine.get_all_boxes()
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

    def test_get_score_2x2(self):
        game_engine = dnbpy.GameEngine((2, 2), ['player1', 'player2'])
        game_engine.select_edge(0, 'player1')
        game_engine.select_edge(2, 'player2')
        game_engine.select_edge(3, 'player1')
        game_engine.select_edge(5, 'player2')
        score = game_engine.get_score('player1')
        self.assertEqual(score, 0)
        score = game_engine.get_score('player2')
        self.assertEqual(score, 1)
        self.assertFalse(game_engine.is_game_finished())
        game_engine.select_edge(11, 'player2')
        game_engine.select_edge(6, 'player1')
        game_engine.select_edge(10, 'player2')
        game_engine.select_edge(8, 'player1')
        score = game_engine.get_score('player1')
        self.assertEqual(score, 0)
        score = game_engine.get_score('player2')
        self.assertEqual(score, 1)
        game_engine.select_edge(9, 'player2')
        game_engine.select_edge(7, 'player2')
        score = game_engine.get_score('player1')
        self.assertEqual(score, 0)
        score = game_engine.get_score('player2')
        self.assertEqual(score, 3)
        game_engine.select_edge(4, 'player2')
        game_engine.select_edge(1, 'player1')
        score = game_engine.get_score('player1')
        self.assertEqual(score, 1)
        score = game_engine.get_score('player2')
        self.assertEqual(score, 3)
        self.assertTrue(game_engine.is_game_finished())
        with self.assertRaises(Exception) as e:
            game_engine.select_edge(1, 'player1')
        self.assertTrue(("game is finished" in str(e.exception)))
        boxes = game_engine.get_boxes('player1')
        self.assertEqual(len(boxes), 1)
        self.assertEqual(str(boxes[0]), '1-3-4-6')
        boxes = game_engine.get_boxes('player2')
        self.assertEqual(len(boxes), 3)
        self.assertEqual(str(boxes[0]), '0-2-3-5')
        self.assertEqual(str(boxes[1]), '6-8-9-11')
        self.assertEqual(str(boxes[2]), '5-7-8-10')

    def test_select_edge_returns_next_player(self):
        game_engine = dnbpy.GameEngine((1, 1), ['player1', 'player2'])
        next_player = game_engine.select_edge(0, 'player1')
        self.assertEqual(next_player, 'player2')
        next_player = game_engine.select_edge(1, 'player2')
        self.assertEqual(next_player, 'player1')
        next_player = game_engine.select_edge(2, 'player1')
        self.assertEqual(next_player, 'player2')
        next_player = game_engine.select_edge(3, 'player2')
        self.assertIsNone(next_player)
        self.assertEqual(game_engine.get_score('player1'), 0)
        self.assertEqual(game_engine.get_score('player2'), 1)
        with self.assertRaises(Exception) as e:
            game_engine.select_edge(1, 'player1')
        self.assertTrue(("game is finished" in str(e.exception)))

    def test_get_score_player_not_recognized(self):
        with self.assertRaises(Exception) as e:
            game_engine = dnbpy.GameEngine((2, 2), ['player1', 'player2'])
            game_engine.get_score('player3')
        self.assertTrue(("player not recognized: player3" in str(e.exception)))

    def test_get_current_player(self):
        game_engine = dnbpy.GameEngine((1, 1), ['player1', 'player2'])
        game_engine.select_edge(0, 'player1')
        self.assertEqual(game_engine.get_current_player(), 'player2')
        game_engine.select_edge(1, 'player2')
        self.assertEqual(game_engine.get_current_player(), 'player1')
        game_engine.select_edge(2, 'player1')
        self.assertEqual(game_engine.get_current_player(), 'player2')
        game_engine.select_edge(3, 'player2')
        self.assertIsNone(game_engine.get_current_player())

    def test_get_boxes_2x2(self):
        game_engine = dnbpy.GameEngine((2, 2), ['player1', 'player2'])
        game_engine.select_edge(0, 'player1')
        game_engine.select_edge(2, 'player2')
        game_engine.select_edge(3, 'player1')
        game_engine.select_edge(5, 'player2')
        boxes = game_engine.get_boxes('player1')
        self.assertEqual(len(boxes), 0)
        boxes = game_engine.get_boxes('player2')
        self.assertEqual(len(boxes), 1)
        self.assertEqual(str(boxes[0]), '0-2-3-5')

    def test_get_boxes_player_not_recognized(self):
        with self.assertRaises(Exception) as e:
            game_engine = dnbpy.GameEngine((2, 2), ['player1', 'player2'])
            game_engine.get_boxes('player3')
        self.assertTrue(("player not recognized: player3" in str(e.exception)))

    def test_get_edge_matrix_3x3(self):
        game_engine = dnbpy.GameEngine((3, 3), ['player1', 'player2'])
        edge_matrix = game_engine.get_edge_matrix()
        self.assertEqual(edge_matrix.tolist(), [[0, 0, 0, 0],
                                                [0, 0, 0, 0],
                                                [0, 0, 0, 0],
                                                [0, 0, 0, 0]])
        game_engine.select_edge(0, 'player1')
        self.assertEqual(edge_matrix.tolist(), [[1, 1, 0, 0],
                                                [0, 0, 0, 0],
                                                [0, 0, 0, 0],
                                                [0, 0, 0, 0]])
        game_engine.select_edge(1, 'player2')
        self.assertEqual(edge_matrix.tolist(), [[1, 2, 1, 0],
                                                [0, 0, 0, 0],
                                                [0, 0, 0, 0],
                                                [0, 0, 0, 0]])
        game_engine.select_edge(4, 'player1')
        self.assertEqual(edge_matrix.tolist(), [[1, 3, 1, 0],
                                                [0, 1, 0, 0],
                                                [0, 0, 0, 0],
                                                [0, 0, 0, 0]])
        game_engine.select_edge(3, 'player2')
        self.assertEqual(edge_matrix.tolist(), [[2, 3, 1, 0],
                                                [1, 1, 0, 0],
                                                [0, 0, 0, 0],
                                                [0, 0, 0, 0]])
        game_engine.select_edge(7, 'player1')
        self.assertEqual(edge_matrix.tolist(), [[2, 3, 1, 0],
                                                [2, 2, 0, 0],
                                                [0, 0, 0, 0],
                                                [0, 0, 0, 0]])
        score = game_engine.get_score('player1')
        self.assertEqual(score, 1)
        score = game_engine.get_score('player2')
        self.assertEqual(score, 0)
        game_engine.select_edge(23, 'player1')
        self.assertEqual(edge_matrix.tolist(), [[2, 3, 1, 0],
                                                [2, 2, 0, 0],
                                                [0, 0, 0, 0],
                                                [0, 0, 1, 1]])
        game_engine.select_edge(20, 'player2')
        self.assertEqual(edge_matrix.tolist(), [[2, 3, 1, 0],
                                                [2, 2, 0, 0],
                                                [0, 0, 0, 1],
                                                [0, 0, 1, 2]])
        game_engine.select_edge(19, 'player1')
        self.assertEqual(edge_matrix.tolist(), [[2, 3, 1, 0],
                                                [2, 2, 0, 0],
                                                [0, 0, 1, 1],
                                                [0, 0, 2, 2]])
        game_engine.select_edge(16, 'player2')
        self.assertEqual(edge_matrix.tolist(), [[2, 3, 1, 0],
                                                [2, 2, 0, 0],
                                                [0, 0, 2, 2],
                                                [0, 0, 2, 2]])
        score = game_engine.get_score('player1')
        self.assertEqual(score, 1)
        score = game_engine.get_score('player2')
        self.assertEqual(score, 1)
        game_engine.select_edge(8, 'player2')
        self.assertEqual(edge_matrix.tolist(), [[2, 3, 1, 0],
                                                [2, 3, 1, 0],
                                                [0, 0, 2, 2],
                                                [0, 0, 2, 2]])
        game_engine.select_edge(11, 'player1')
        self.assertEqual(edge_matrix.tolist(), [[2, 3, 1, 0],
                                                [2, 4, 1, 0],
                                                [0, 1, 2, 2],
                                                [0, 0, 2, 2]])
        game_engine.select_edge(5, 'player2')
        self.assertEqual(edge_matrix.tolist(), [[2, 3, 2, 0],
                                                [2, 4, 2, 0],
                                                [0, 1, 2, 2],
                                                [0, 0, 2, 2]])
        score = game_engine.get_score('player1')
        self.assertEqual(score, 1)
        score = game_engine.get_score('player2')
        self.assertEqual(score, 2)
        boxes = game_engine.get_boxes('player1')
        self.assertEqual(len(boxes), 1)
        self.assertEqual(str(boxes[0]), '0-3-4-7')
        boxes = game_engine.get_boxes('player2')
        self.assertEqual(len(boxes), 2)
        self.assertEqual(str(boxes[0]), '16-19-20-23')
        self.assertEqual(str(boxes[1]), '1-4-5-8')

    def test_convert_string_index_to_coordinates_1x1(self):
        game_engine = dnbpy.GameEngine((1, 1), ['player1', 'player2'])
        coordinates = game_engine.convert_string_index_to_coordinates(0)
        self.assertEqual(coordinates, ((0, 0), (0, 1)))
        coordinates = game_engine.convert_string_index_to_coordinates(1)
        self.assertEqual(coordinates, ((0, 0), (1, 0)))
        coordinates = game_engine.convert_string_index_to_coordinates(2)
        self.assertEqual(coordinates, ((0, 1), (1, 1)))
        coordinates = game_engine.convert_string_index_to_coordinates(3)
        self.assertEqual(coordinates, ((1, 0), (1, 1)))

    def test_convert_string_index_to_coordinates_1x2(self):
        game_engine = dnbpy.GameEngine((1, 2), ['player1', 'player2'])
        coordinates = game_engine.convert_string_index_to_coordinates(0)
        self.assertEqual(coordinates, ((0, 0), (0, 1)))
        coordinates = game_engine.convert_string_index_to_coordinates(1)
        self.assertEqual(coordinates, ((0, 1), (0, 2)))
        coordinates = game_engine.convert_string_index_to_coordinates(2)
        self.assertEqual(coordinates, ((0, 0), (1, 0)))
        coordinates = game_engine.convert_string_index_to_coordinates(3)
        self.assertEqual(coordinates, ((0, 1), (1, 1)))
        coordinates = game_engine.convert_string_index_to_coordinates(4)
        self.assertEqual(coordinates, ((0, 2), (1, 2)))
        coordinates = game_engine.convert_string_index_to_coordinates(5)
        self.assertEqual(coordinates, ((1, 0), (1, 1)))
        coordinates = game_engine.convert_string_index_to_coordinates(6)
        self.assertEqual(coordinates, ((1, 1), (1, 2)))

    def test_convert_string_index_to_coordinates_2x1(self):
        game_engine = dnbpy.GameEngine((2, 1), ['player1', 'player2'])
        coordinates = game_engine.convert_string_index_to_coordinates(0)
        self.assertEqual(coordinates, ((0, 0), (0, 1)))
        coordinates = game_engine.convert_string_index_to_coordinates(1)
        self.assertEqual(coordinates, ((0, 0), (1, 0)))
        coordinates = game_engine.convert_string_index_to_coordinates(2)
        self.assertEqual(coordinates, ((0, 1), (1, 1)))
        coordinates = game_engine.convert_string_index_to_coordinates(3)
        self.assertEqual(coordinates, ((1, 0), (1, 1)))
        coordinates = game_engine.convert_string_index_to_coordinates(4)
        self.assertEqual(coordinates, ((1, 0), (2, 0)))
        coordinates = game_engine.convert_string_index_to_coordinates(5)
        self.assertEqual(coordinates, ((1, 1), (2, 1)))
        coordinates = game_engine.convert_string_index_to_coordinates(6)
        self.assertEqual(coordinates, ((2, 0), (2, 1)))

    def test_convert_string_index_to_coordinates_2x2(self):
        game_engine = dnbpy.GameEngine((2, 2), ['player1', 'player2'])
        coordinates = game_engine.convert_string_index_to_coordinates(0)
        self.assertEqual(coordinates, ((0, 0), (0, 1)))
        coordinates = game_engine.convert_string_index_to_coordinates(1)
        self.assertEqual(coordinates, ((0, 1), (0, 2)))
        coordinates = game_engine.convert_string_index_to_coordinates(2)
        self.assertEqual(coordinates, ((0, 0), (1, 0)))
        coordinates = game_engine.convert_string_index_to_coordinates(3)
        self.assertEqual(coordinates, ((0, 1), (1, 1)))
        coordinates = game_engine.convert_string_index_to_coordinates(4)
        self.assertEqual(coordinates, ((0, 2), (1, 2)))
        coordinates = game_engine.convert_string_index_to_coordinates(5)
        self.assertEqual(coordinates, ((1, 0), (1, 1)))
        coordinates = game_engine.convert_string_index_to_coordinates(6)
        self.assertEqual(coordinates, ((1, 1), (1, 2)))
        coordinates = game_engine.convert_string_index_to_coordinates(7)
        self.assertEqual(coordinates, ((1, 0), (2, 0)))
        coordinates = game_engine.convert_string_index_to_coordinates(8)
        self.assertEqual(coordinates, ((1, 1), (2, 1)))
        coordinates = game_engine.convert_string_index_to_coordinates(9)
        self.assertEqual(coordinates, ((1, 2), (2, 2)))
        coordinates = game_engine.convert_string_index_to_coordinates(10)
        self.assertEqual(coordinates, ((2, 0), (2, 1)))
        coordinates = game_engine.convert_string_index_to_coordinates(11)
        self.assertEqual(coordinates, ((2, 1), (2, 2)))

    def test_convert_string_index_to_coordinates_3x3(self):
        game_engine = dnbpy.GameEngine((3, 3), ['player1', 'player2'])
        coordinates = game_engine.convert_string_index_to_coordinates(0)
        self.assertEqual(coordinates, ((0, 0), (0, 1)))
        coordinates = game_engine.convert_string_index_to_coordinates(1)
        self.assertEqual(coordinates, ((0, 1), (0, 2)))
        coordinates = game_engine.convert_string_index_to_coordinates(2)
        self.assertEqual(coordinates, ((0, 2), (0, 3)))
        coordinates = game_engine.convert_string_index_to_coordinates(3)
        self.assertEqual(coordinates, ((0, 0), (1, 0)))
        coordinates = game_engine.convert_string_index_to_coordinates(4)
        self.assertEqual(coordinates, ((0, 1), (1, 1)))
        coordinates = game_engine.convert_string_index_to_coordinates(5)
        self.assertEqual(coordinates, ((0, 2), (1, 2)))
        coordinates = game_engine.convert_string_index_to_coordinates(6)
        self.assertEqual(coordinates, ((0, 3), (1, 3)))
        coordinates = game_engine.convert_string_index_to_coordinates(7)
        self.assertEqual(coordinates, ((1, 0), (1, 1)))
        coordinates = game_engine.convert_string_index_to_coordinates(8)
        self.assertEqual(coordinates, ((1, 1), (1, 2)))
        coordinates = game_engine.convert_string_index_to_coordinates(9)
        self.assertEqual(coordinates, ((1, 2), (1, 3)))
        coordinates = game_engine.convert_string_index_to_coordinates(10)
        self.assertEqual(coordinates, ((1, 0), (2, 0)))
        coordinates = game_engine.convert_string_index_to_coordinates(11)
        self.assertEqual(coordinates, ((1, 1), (2, 1)))
        coordinates = game_engine.convert_string_index_to_coordinates(12)
        self.assertEqual(coordinates, ((1, 2), (2, 2)))
        coordinates = game_engine.convert_string_index_to_coordinates(13)
        self.assertEqual(coordinates, ((1, 3), (2, 3)))
        coordinates = game_engine.convert_string_index_to_coordinates(14)
        self.assertEqual(coordinates, ((2, 0), (2, 1)))
        coordinates = game_engine.convert_string_index_to_coordinates(15)
        self.assertEqual(coordinates, ((2, 1), (2, 2)))
        coordinates = game_engine.convert_string_index_to_coordinates(16)
        self.assertEqual(coordinates, ((2, 2), (2, 3)))
        coordinates = game_engine.convert_string_index_to_coordinates(17)
        self.assertEqual(coordinates, ((2, 0), (3, 0)))
        coordinates = game_engine.convert_string_index_to_coordinates(18)
        self.assertEqual(coordinates, ((2, 1), (3, 1)))
        coordinates = game_engine.convert_string_index_to_coordinates(19)
        self.assertEqual(coordinates, ((2, 2), (3, 2)))
        coordinates = game_engine.convert_string_index_to_coordinates(20)
        self.assertEqual(coordinates, ((2, 3), (3, 3)))
        coordinates = game_engine.convert_string_index_to_coordinates(21)
        self.assertEqual(coordinates, ((3, 0), (3, 1)))
        coordinates = game_engine.convert_string_index_to_coordinates(22)
        self.assertEqual(coordinates, ((3, 1), (3, 2)))
        coordinates = game_engine.convert_string_index_to_coordinates(23)
        self.assertEqual(coordinates, ((3, 2), (3, 3)))

    # TODO game_engine to string:
    """
    *    0    *    1    * 

    2         3         4
    
    *    5    *    6    *
    
    7         8         9
    
    *    10   *    11   * 
    player1: 0, p2: 0
    
    
    *    0    *    -    * 
    
    2         | player1 |
    
    *    5    *    -    *
    
    |         |    p2   |
    
    *    10   *    -    * 
    player1: 1, p2: 1
    
    
    *    0    *    1    *     2    *
    
    3         4         5          6
    
    *    7    *    8    *     9    *
    
    10        11        12         13
    
    *    14   *    15   *     16   *
    
    17        18        19         20
    
    *    21   *    22   *     23   *
    player1: 0, p2: 0
    """
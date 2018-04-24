import unittest

import dnbpy


class TestDNBPy(unittest.TestCase):

    # TODO test constructor args are valid

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

    # TODO test select_edge player not recognized throws exception
    # TODO test select_edge index does not exist throws exception
    # TODO test select_edge index already taken throws exception

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

    # TODO test get_score player not recognized throws exception

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

    # TODO test get_boxes player not recognized throws exception
import unittest

import dnbpy


class TestGetEdgeMatrix(unittest.TestCase):

    def test_get_edge_matrix_1x1(self):
        game = dnbpy.Game((1, 1), ['player1', 'player2'])
        edge_matrix = game.get_edge_matrix()
        self.assertEqual(edge_matrix.tolist(), [[1, 0, 1],
                                                [0, 0, 0],
                                                [1, 0, 1]])
        game.select_edge(0, 'player1')
        edge_matrix = game.get_edge_matrix()
        self.assertEqual(edge_matrix.tolist(), [[1, 1, 1],
                                                [0, 0, 0],
                                                [1, 0, 1]])
        game.select_edge(2, 'player2')
        edge_matrix = game.get_edge_matrix()
        self.assertEqual(edge_matrix.tolist(), [[1, 1, 1],
                                                [0, 0, 1],
                                                [1, 0, 1]])
        game.select_edge(1, 'player1')
        edge_matrix = game.get_edge_matrix()
        self.assertEqual(edge_matrix.tolist(), [[1, 1, 1],
                                                [1, 0, 1],
                                                [1, 0, 1]])
        game.select_edge(3, 'player2')
        edge_matrix = game.get_edge_matrix()
        self.assertEqual(edge_matrix.tolist(), [[1, 1, 1],
                                                [1, 0, 1],
                                                [1, 1, 1]])
        score = game.get_score('player1')
        self.assertEqual(score, 0)
        score = game.get_score('player2')
        self.assertEqual(score, 1)
        boxes = game.get_boxes('player1')
        self.assertEqual(len(boxes), 0)
        boxes = game.get_boxes('player2')
        self.assertEqual(len(boxes), 1)
        self.assertEqual(str(boxes[0]), '0-1-2-3')
        self.assertTrue(game.is_finished())

    def test_get_edge_matrix_1x2(self):
        game = dnbpy.Game((1, 2), ['player1', 'player2'])
        edge_matrix = game.get_edge_matrix()
        self.assertEqual(edge_matrix.tolist(), [[1, 0, 1, 0, 1],
                                                [0, 0, 0, 0, 0],
                                                [1, 0, 1, 0, 1]])
        game.select_edge(0, 'player1')
        edge_matrix = game.get_edge_matrix()
        self.assertEqual(edge_matrix.tolist(), [[1, 1, 1, 0, 1],
                                                [0, 0, 0, 0, 0],
                                                [1, 0, 1, 0, 1]])
        game.select_edge(2, 'player2')
        edge_matrix = game.get_edge_matrix()
        self.assertEqual(edge_matrix.tolist(), [[1, 1, 1, 0, 1],
                                                [1, 0, 0, 0, 0],
                                                [1, 0, 1, 0, 1]])
        game.select_edge(1, 'player1')
        edge_matrix = game.get_edge_matrix()
        self.assertEqual(edge_matrix.tolist(), [[1, 1, 1, 1, 1],
                                                [1, 0, 0, 0, 0],
                                                [1, 0, 1, 0, 1]])
        game.select_edge(3, 'player2')
        edge_matrix = game.get_edge_matrix()
        self.assertEqual(edge_matrix.tolist(), [[1, 1, 1, 1, 1],
                                                [1, 0, 1, 0, 0],
                                                [1, 0, 1, 0, 1]])
        game.select_edge(4, 'player1')
        edge_matrix = game.get_edge_matrix()
        self.assertEqual(edge_matrix.tolist(), [[1, 1, 1, 1, 1],
                                                [1, 0, 1, 0, 1],
                                                [1, 0, 1, 0, 1]])
        game.select_edge(5, 'player2')
        edge_matrix = game.get_edge_matrix()
        self.assertEqual(edge_matrix.tolist(), [[1, 1, 1, 1, 1],
                                                [1, 0, 1, 0, 1],
                                                [1, 1, 1, 0, 1]])
        game.select_edge(6, 'player2')
        edge_matrix = game.get_edge_matrix()
        self.assertEqual(edge_matrix.tolist(), [[1, 1, 1, 1, 1],
                                                [1, 0, 1, 0, 1],
                                                [1, 1, 1, 1, 1]])
        score = game.get_score('player1')
        self.assertEqual(score, 0)
        score = game.get_score('player2')
        self.assertEqual(score, 2)
        boxes = game.get_boxes('player1')
        self.assertEqual(len(boxes), 0)
        boxes = game.get_boxes('player2')
        self.assertEqual(len(boxes), 2)
        self.assertEqual(str(boxes[0]), '0-2-3-5')
        self.assertEqual(str(boxes[1]), '1-3-4-6')
        self.assertTrue(game.is_finished())

    def test_get_edge_matrix_2x1(self):
        game = dnbpy.Game((2, 1), ['player1', 'player2'])
        edge_matrix = game.get_edge_matrix()
        self.assertEqual(edge_matrix.tolist(), [[1, 0, 1],
                                                [0, 0, 0],
                                                [1, 0, 1],
                                                [0, 0, 0],
                                                [1, 0, 1]])
        game.select_edge(0, 'player1')
        edge_matrix = game.get_edge_matrix()
        self.assertEqual(edge_matrix.tolist(), [[1, 1, 1],
                                                [0, 0, 0],
                                                [1, 0, 1],
                                                [0, 0, 0],
                                                [1, 0, 1]])
        game.select_edge(2, 'player2')
        edge_matrix = game.get_edge_matrix()
        self.assertEqual(edge_matrix.tolist(), [[1, 1, 1],
                                                [0, 0, 1],
                                                [1, 0, 1],
                                                [0, 0, 0],
                                                [1, 0, 1]])
        game.select_edge(1, 'player1')
        edge_matrix = game.get_edge_matrix()
        self.assertEqual(edge_matrix.tolist(), [[1, 1, 1],
                                                [1, 0, 1],
                                                [1, 0, 1],
                                                [0, 0, 0],
                                                [1, 0, 1]])
        game.select_edge(3, 'player2')
        edge_matrix = game.get_edge_matrix()
        self.assertEqual(edge_matrix.tolist(), [[1, 1, 1],
                                                [1, 0, 1],
                                                [1, 1, 1],
                                                [0, 0, 0],
                                                [1, 0, 1]])
        game.select_edge(5, 'player2')
        edge_matrix = game.get_edge_matrix()
        self.assertEqual(edge_matrix.tolist(), [[1, 1, 1],
                                                [1, 0, 1],
                                                [1, 1, 1],
                                                [0, 0, 1],
                                                [1, 0, 1]])
        game.select_edge(6, 'player1')
        edge_matrix = game.get_edge_matrix()
        self.assertEqual(edge_matrix.tolist(), [[1, 1, 1],
                                                [1, 0, 1],
                                                [1, 1, 1],
                                                [0, 0, 1],
                                                [1, 1, 1]])
        game.select_edge(4, 'player2')
        edge_matrix = game.get_edge_matrix()
        self.assertEqual(edge_matrix.tolist(), [[1, 1, 1],
                                                [1, 0, 1],
                                                [1, 1, 1],
                                                [1, 0, 1],
                                                [1, 1, 1]])
        score = game.get_score('player1')
        self.assertEqual(score, 0)
        score = game.get_score('player2')
        self.assertEqual(score, 2)
        boxes = game.get_boxes('player1')
        self.assertEqual(len(boxes), 0)
        boxes = game.get_boxes('player2')
        self.assertEqual(len(boxes), 2)
        self.assertEqual(str(boxes[0]), '0-1-2-3')
        self.assertEqual(str(boxes[1]), '3-4-5-6')
        self.assertTrue(game.is_finished())

    def test_get_edge_matrix_2x2(self):
        game = dnbpy.Game((2, 2), ['player1', 'player2'])
        edge_matrix = game.get_edge_matrix()
        self.assertEqual(edge_matrix.tolist(), [[1, 0, 1, 0, 1],
                                                [0, 0, 0, 0, 0],
                                                [1, 0, 1, 0, 1],
                                                [0, 0, 0, 0, 0],
                                                [1, 0, 1, 0, 1]])
        game.select_edge(6, 'player1')
        edge_matrix = game.get_edge_matrix()
        self.assertEqual(edge_matrix.tolist(), [[1, 0, 1, 0, 1],
                                                [0, 0, 0, 0, 0],
                                                [1, 0, 1, 1, 1],
                                                [0, 0, 0, 0, 0],
                                                [1, 0, 1, 0, 1]])

    def test_get_edge_matrix_3x3(self):
        game = dnbpy.Game((3, 3), ['player1', 'player2'])
        edge_matrix = game.get_edge_matrix()
        self.assertEqual(edge_matrix.tolist(), [[1, 0, 1, 0, 1, 0, 1],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [1, 0, 1, 0, 1, 0, 1],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [1, 0, 1, 0, 1, 0, 1],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [1, 0, 1, 0, 1, 0, 1]])
        game.select_edge(0, 'player1')
        edge_matrix = game.get_edge_matrix()
        self.assertEqual(edge_matrix.tolist(), [[1, 1, 1, 0, 1, 0, 1],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [1, 0, 1, 0, 1, 0, 1],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [1, 0, 1, 0, 1, 0, 1],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [1, 0, 1, 0, 1, 0, 1]])
        game.select_edge(1, 'player2')
        edge_matrix = game.get_edge_matrix()
        self.assertEqual(edge_matrix.tolist(), [[1, 1, 1, 1, 1, 0, 1],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [1, 0, 1, 0, 1, 0, 1],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [1, 0, 1, 0, 1, 0, 1],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [1, 0, 1, 0, 1, 0, 1]])
        game.select_edge(4, 'player1')
        edge_matrix = game.get_edge_matrix()
        self.assertEqual(edge_matrix.tolist(), [[1, 1, 1, 1, 1, 0, 1],
                                                [0, 0, 1, 0, 0, 0, 0],
                                                [1, 0, 1, 0, 1, 0, 1],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [1, 0, 1, 0, 1, 0, 1],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [1, 0, 1, 0, 1, 0, 1]])
        game.select_edge(3, 'player2')
        edge_matrix = game.get_edge_matrix()
        self.assertEqual(edge_matrix.tolist(), [[1, 1, 1, 1, 1, 0, 1],
                                                [1, 0, 1, 0, 0, 0, 0],
                                                [1, 0, 1, 0, 1, 0, 1],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [1, 0, 1, 0, 1, 0, 1],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [1, 0, 1, 0, 1, 0, 1]])
        game.select_edge(7, 'player1')
        edge_matrix = game.get_edge_matrix()
        self.assertEqual(edge_matrix.tolist(), [[1, 1, 1, 1, 1, 0, 1],
                                                [1, 0, 1, 0, 0, 0, 0],
                                                [1, 1, 1, 0, 1, 0, 1],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [1, 0, 1, 0, 1, 0, 1],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [1, 0, 1, 0, 1, 0, 1]])
        score = game.get_score('player1')
        self.assertEqual(score, 1)
        score = game.get_score('player2')
        self.assertEqual(score, 0)
        game.select_edge(23, 'player1')
        edge_matrix = game.get_edge_matrix()
        self.assertEqual(edge_matrix.tolist(), [[1, 1, 1, 1, 1, 0, 1],
                                                [1, 0, 1, 0, 0, 0, 0],
                                                [1, 1, 1, 0, 1, 0, 1],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [1, 0, 1, 0, 1, 0, 1],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [1, 0, 1, 0, 1, 1, 1]])
        game.select_edge(20, 'player2')
        edge_matrix = game.get_edge_matrix()
        self.assertEqual(edge_matrix.tolist(), [[1, 1, 1, 1, 1, 0, 1],
                                                [1, 0, 1, 0, 0, 0, 0],
                                                [1, 1, 1, 0, 1, 0, 1],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [1, 0, 1, 0, 1, 0, 1],
                                                [0, 0, 0, 0, 0, 0, 1],
                                                [1, 0, 1, 0, 1, 1, 1]])
        game.select_edge(19, 'player1')
        edge_matrix = game.get_edge_matrix()
        self.assertEqual(edge_matrix.tolist(), [[1, 1, 1, 1, 1, 0, 1],
                                                [1, 0, 1, 0, 0, 0, 0],
                                                [1, 1, 1, 0, 1, 0, 1],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [1, 0, 1, 0, 1, 0, 1],
                                                [0, 0, 0, 0, 1, 0, 1],
                                                [1, 0, 1, 0, 1, 1, 1]])
        game.select_edge(16, 'player2')
        edge_matrix = game.get_edge_matrix()
        self.assertEqual(edge_matrix.tolist(), [[1, 1, 1, 1, 1, 0, 1],
                                                [1, 0, 1, 0, 0, 0, 0],
                                                [1, 1, 1, 0, 1, 0, 1],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [1, 0, 1, 0, 1, 1, 1],
                                                [0, 0, 0, 0, 1, 0, 1],
                                                [1, 0, 1, 0, 1, 1, 1]])
        score = game.get_score('player1')
        self.assertEqual(score, 1)
        score = game.get_score('player2')
        self.assertEqual(score, 1)
        game.select_edge(8, 'player2')
        edge_matrix = game.get_edge_matrix()
        self.assertEqual(edge_matrix.tolist(), [[1, 1, 1, 1, 1, 0, 1],
                                                [1, 0, 1, 0, 0, 0, 0],
                                                [1, 1, 1, 1, 1, 0, 1],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [1, 0, 1, 0, 1, 1, 1],
                                                [0, 0, 0, 0, 1, 0, 1],
                                                [1, 0, 1, 0, 1, 1, 1]])
        game.select_edge(11, 'player1')
        edge_matrix = game.get_edge_matrix()
        self.assertEqual(edge_matrix.tolist(), [[1, 1, 1, 1, 1, 0, 1],
                                                [1, 0, 1, 0, 0, 0, 0],
                                                [1, 1, 1, 1, 1, 0, 1],
                                                [0, 0, 1, 0, 0, 0, 0],
                                                [1, 0, 1, 0, 1, 1, 1],
                                                [0, 0, 0, 0, 1, 0, 1],
                                                [1, 0, 1, 0, 1, 1, 1]])
        game.select_edge(5, 'player2')
        edge_matrix = game.get_edge_matrix()
        self.assertEqual(edge_matrix.tolist(), [[1, 1, 1, 1, 1, 0, 1],
                                                [1, 0, 1, 0, 1, 0, 0],
                                                [1, 1, 1, 1, 1, 0, 1],
                                                [0, 0, 1, 0, 0, 0, 0],
                                                [1, 0, 1, 0, 1, 1, 1],
                                                [0, 0, 0, 0, 1, 0, 1],
                                                [1, 0, 1, 0, 1, 1, 1]])
        score = game.get_score('player1')
        self.assertEqual(score, 1)
        score = game.get_score('player2')
        self.assertEqual(score, 2)
        boxes = game.get_boxes('player1')
        self.assertEqual(len(boxes), 1)
        self.assertEqual(str(boxes[0]), '0-3-4-7')
        boxes = game.get_boxes('player2')
        self.assertEqual(len(boxes), 2)
        self.assertEqual(str(boxes[0]), '16-19-20-23')
        self.assertEqual(str(boxes[1]), '1-4-5-8')

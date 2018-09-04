import unittest

import dnbpy


class TestBoardStateToEdgeMatrix(unittest.TestCase):

    def test_convert_board_state_to_edge_matrix_1x1(self):
        edge_matrix = dnbpy.convert_board_state_to_edge_matrix((1, 1), [0, 0, 0, 0])
        self.assertEqual(edge_matrix, [[1, 0, 1],
                                       [0, 0, 0],
                                       [1, 0, 1]])

        edge_matrix = dnbpy.convert_board_state_to_edge_matrix((1, 1), [1, 0, 0, 0])
        self.assertEqual(edge_matrix, [[1, 1, 1],
                                       [0, 0, 0],
                                       [1, 0, 1]])

        edge_matrix = dnbpy.convert_board_state_to_edge_matrix((1, 1), [0, 1, 0, 0])
        self.assertEqual(edge_matrix, [[1, 0, 1],
                                       [1, 0, 0],
                                       [1, 0, 1]])

        edge_matrix = dnbpy.convert_board_state_to_edge_matrix((1, 1), [0, 0, 1, 0])
        self.assertEqual(edge_matrix, [[1, 0, 1],
                                       [0, 0, 1],
                                       [1, 0, 1]])

        edge_matrix = dnbpy.convert_board_state_to_edge_matrix((1, 1), [0, 0, 0, 1])
        self.assertEqual(edge_matrix, [[1, 0, 1],
                                       [0, 0, 0],
                                       [1, 1, 1]])

        edge_matrix = dnbpy.convert_board_state_to_edge_matrix((1, 1), [1, 1, 0, 0])
        self.assertEqual(edge_matrix, [[1, 1, 1],
                                       [1, 0, 0],
                                       [1, 0, 1]])

        edge_matrix = dnbpy.convert_board_state_to_edge_matrix((1, 1), [1, 0, 1, 0])
        self.assertEqual(edge_matrix, [[1, 1, 1],
                                       [0, 0, 1],
                                       [1, 0, 1]])

        edge_matrix = dnbpy.convert_board_state_to_edge_matrix((1, 1), [1, 0, 0, 1])
        self.assertEqual(edge_matrix, [[1, 1, 1],
                                       [0, 0, 0],
                                       [1, 1, 1]])

        edge_matrix = dnbpy.convert_board_state_to_edge_matrix((1, 1), [1, 1, 1, 1])
        self.assertEqual(edge_matrix, [[1, 1, 1],
                                       [1, 0, 1],
                                       [1, 1, 1]])

    def test_convert_board_state_to_edge_matrix_1x1_edge_length(self):
        edge_matrix = dnbpy.convert_board_state_to_edge_matrix((1, 1), [0, 0, 0, 0], edge_length=2)
        self.assertEqual(edge_matrix, [[1, 0, 0, 1],
                                       [0, 0, 0, 0],
                                       [0, 0, 0, 0],
                                       [1, 0, 0, 1]])

        edge_matrix = dnbpy.convert_board_state_to_edge_matrix((1, 1), [0, 1, 0, 0], edge_length=2)
        self.assertEqual(edge_matrix, [[1, 0, 0, 1],
                                       [1, 0, 0, 0],
                                       [1, 0, 0, 0],
                                       [1, 0, 0, 1]])

    def test_convert_board_state_to_edge_matrix_1x1_edge_length_include_dots(self):
        edge_matrix = dnbpy.convert_board_state_to_edge_matrix((1, 1), [0, 0, 0, 0], edge_length=2, include_dots=False)
        self.assertEqual(edge_matrix, [[0, 0, 0, 0],
                                       [0, 0, 0, 0],
                                       [0, 0, 0, 0],
                                       [0, 0, 0, 0]])

        edge_matrix = dnbpy.convert_board_state_to_edge_matrix((1, 1), [0, 1, 0, 0], edge_length=2, include_dots=False)
        self.assertEqual(edge_matrix, [[1, 0, 0, 0],
                                       [1, 0, 0, 0],
                                       [1, 0, 0, 0],
                                       [1, 0, 0, 0]])

    def test_convert_board_state_to_edge_matrix_2x2_edge_length(self):
        edge_matrix = dnbpy.convert_board_state_to_edge_matrix((2, 2), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                               edge_length=2)
        self.assertEqual(edge_matrix, [[1, 0, 0, 1, 0, 0, 1],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [1, 0, 0, 1, 0, 0, 1],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [1, 0, 0, 1, 0, 0, 1]])

        edge_matrix = dnbpy.convert_board_state_to_edge_matrix((2, 2), [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                                                               edge_length=2)
        self.assertEqual(edge_matrix, [[1, 0, 0, 1, 1, 1, 1],
                                       [0, 0, 0, 1, 0, 0, 0],
                                       [0, 0, 0, 1, 0, 0, 0],
                                       [1, 0, 0, 1, 0, 0, 1],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [1, 0, 0, 1, 1, 1, 1]])

    def test_convert_board_state_to_edge_matrix_2x2_edge_length_include_dots(self):
        edge_matrix = dnbpy.convert_board_state_to_edge_matrix((2, 2), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                               edge_length=2, include_dots=False)
        self.assertEqual(edge_matrix, [[0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0]])

        edge_matrix = dnbpy.convert_board_state_to_edge_matrix((2, 2), [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                                                               edge_length=2, include_dots=False)
        self.assertEqual(edge_matrix, [[0, 0, 0, 1, 1, 1, 1],
                                       [0, 0, 0, 1, 0, 0, 0],
                                       [0, 0, 0, 1, 0, 0, 0],
                                       [0, 0, 0, 1, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [0, 0, 0, 1, 1, 1, 1]])

    def test_convert_board_state_to_edge_matrix_3x3(self):
        edge_matrix = dnbpy.convert_board_state_to_edge_matrix((3, 3), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(edge_matrix, [[1, 0, 1, 0, 1, 0, 1],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [1, 0, 1, 0, 1, 0, 1],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [1, 0, 1, 0, 1, 0, 1],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [1, 0, 1, 0, 1, 0, 1]])

        edge_matrix = dnbpy.convert_board_state_to_edge_matrix((3, 3), [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(edge_matrix, [[1, 1, 1, 0, 1, 0, 1],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [1, 0, 1, 0, 1, 0, 1],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [1, 0, 1, 0, 1, 0, 1],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [1, 0, 1, 0, 1, 0, 1]])

        edge_matrix = dnbpy.convert_board_state_to_edge_matrix((3, 3), [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(edge_matrix, [[1, 1, 1, 1, 1, 0, 1],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [1, 0, 1, 0, 1, 0, 1],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [1, 0, 1, 0, 1, 0, 1],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [1, 0, 1, 0, 1, 0, 1]])

        edge_matrix = dnbpy.convert_board_state_to_edge_matrix((3, 3), [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                                                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(edge_matrix, [[1, 1, 1, 1, 1, 0, 1],
                                       [0, 0, 1, 0, 0, 0, 0],
                                       [1, 0, 1, 0, 1, 0, 1],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [1, 0, 1, 0, 1, 0, 1],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [1, 0, 1, 0, 1, 0, 1]])

        edge_matrix = dnbpy.convert_board_state_to_edge_matrix((3, 3), [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                                                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(edge_matrix, [[1, 1, 1, 1, 1, 0, 1],
                                       [1, 0, 1, 0, 0, 0, 0],
                                       [1, 0, 1, 0, 1, 0, 1],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [1, 0, 1, 0, 1, 0, 1],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [1, 0, 1, 0, 1, 0, 1]])

        edge_matrix = dnbpy.convert_board_state_to_edge_matrix((3, 3), [1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0,
                                                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(edge_matrix, [[1, 1, 1, 1, 1, 0, 1],
                                       [1, 0, 1, 0, 0, 0, 0],
                                       [1, 1, 1, 0, 1, 0, 1],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [1, 0, 1, 0, 1, 0, 1],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [1, 0, 1, 0, 1, 0, 1]])

        edge_matrix = dnbpy.convert_board_state_to_edge_matrix((3, 3), [1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0,
                                                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        self.assertEqual(edge_matrix, [[1, 1, 1, 1, 1, 0, 1],
                                       [1, 0, 1, 0, 0, 0, 0],
                                       [1, 1, 1, 0, 1, 0, 1],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [1, 0, 1, 0, 1, 0, 1],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [1, 0, 1, 0, 1, 1, 1]])

        edge_matrix = dnbpy.convert_board_state_to_edge_matrix((3, 3), [1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0,
                                                                        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1])
        self.assertEqual(edge_matrix, [[1, 1, 1, 1, 1, 0, 1],
                                       [1, 0, 1, 0, 0, 0, 0],
                                       [1, 1, 1, 0, 1, 0, 1],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [1, 0, 1, 0, 1, 0, 1],
                                       [0, 0, 0, 0, 0, 0, 1],
                                       [1, 0, 1, 0, 1, 1, 1]])

        edge_matrix = dnbpy.convert_board_state_to_edge_matrix((3, 3), [1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0,
                                                                        0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1])
        self.assertEqual(edge_matrix, [[1, 1, 1, 1, 1, 0, 1],
                                       [1, 0, 1, 0, 0, 0, 0],
                                       [1, 1, 1, 0, 1, 0, 1],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [1, 0, 1, 0, 1, 0, 1],
                                       [0, 0, 0, 0, 1, 0, 1],
                                       [1, 0, 1, 0, 1, 1, 1]])

        edge_matrix = dnbpy.convert_board_state_to_edge_matrix((3, 3), [1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0,
                                                                        0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1])
        self.assertEqual(edge_matrix, [[1, 1, 1, 1, 1, 0, 1],
                                       [1, 0, 1, 0, 0, 0, 0],
                                       [1, 1, 1, 0, 1, 0, 1],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [1, 0, 1, 0, 1, 1, 1],
                                       [0, 0, 0, 0, 1, 0, 1],
                                       [1, 0, 1, 0, 1, 1, 1]])

        edge_matrix = dnbpy.convert_board_state_to_edge_matrix((3, 3), [1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0,
                                                                        0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1])
        self.assertEqual(edge_matrix, [[1, 1, 1, 1, 1, 0, 1],
                                       [1, 0, 1, 0, 0, 0, 0],
                                       [1, 1, 1, 1, 1, 0, 1],
                                       [0, 0, 0, 0, 0, 0, 0],
                                       [1, 0, 1, 0, 1, 1, 1],
                                       [0, 0, 0, 0, 1, 0, 1],
                                       [1, 0, 1, 0, 1, 1, 1]])

        edge_matrix = dnbpy.convert_board_state_to_edge_matrix((3, 3), [1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1,
                                                                        0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1])
        self.assertEqual(edge_matrix, [[1, 1, 1, 1, 1, 0, 1],
                                       [1, 0, 1, 0, 0, 0, 0],
                                       [1, 1, 1, 1, 1, 0, 1],
                                       [0, 0, 1, 0, 0, 0, 0],
                                       [1, 0, 1, 0, 1, 1, 1],
                                       [0, 0, 0, 0, 1, 0, 1],
                                       [1, 0, 1, 0, 1, 1, 1]])

        edge_matrix = dnbpy.convert_board_state_to_edge_matrix((3, 3), [1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1,
                                                                        0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1])
        self.assertEqual(edge_matrix, [[1, 1, 1, 1, 1, 0, 1],
                                       [1, 0, 1, 0, 1, 0, 0],
                                       [1, 1, 1, 1, 1, 0, 1],
                                       [0, 0, 1, 0, 0, 0, 0],
                                       [1, 0, 1, 0, 1, 1, 1],
                                       [0, 0, 0, 0, 1, 0, 1],
                                       [1, 0, 1, 0, 1, 1, 1]])

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

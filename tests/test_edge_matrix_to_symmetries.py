import unittest

import dnbpy


class TestGetEdgeMatrix(unittest.TestCase):

    def test_convert_edge_matrix_to_symmetries_1x1(self):
        edge_matrix = [[1, 0, 1],
                       [0, 0, 0],
                       [1, 0, 1]]
        symmetries = dnbpy.convert_edge_matrix_to_symmetries(edge_matrix)
        self.assertEqual(len(symmetries), 1)
        self.assertEqual(symmetries[0], [[1, 0, 1],
                                         [0, 0, 0],
                                         [1, 0, 1]])

        edge_matrix = [[1, 1, 1],
                       [0, 0, 0],
                       [1, 0, 1]]
        symmetries = dnbpy.convert_edge_matrix_to_symmetries(edge_matrix)
        self.assertEqual(len(symmetries), 4)
        self.assertEqual(symmetries[0], [[1, 1, 1],
                                         [0, 0, 0],
                                         [1, 0, 1]])
        self.assertEqual(symmetries[1], [[1, 0, 1],
                                         [1, 0, 0],
                                         [1, 0, 1]])
        self.assertEqual(symmetries[2], [[1, 0, 1],
                                         [0, 0, 0],
                                         [1, 1, 1]])
        self.assertEqual(symmetries[3], [[1, 0, 1],
                                         [0, 0, 1],
                                         [1, 0, 1]])

        edge_matrix = [[1, 1, 1],
                       [0, 0, 0],
                       [1, 1, 1]]
        symmetries = dnbpy.convert_edge_matrix_to_symmetries(edge_matrix)
        self.assertEqual(len(symmetries), 2)
        self.assertEqual(symmetries[0], [[1, 1, 1],
                                         [0, 0, 0],
                                         [1, 1, 1]])
        self.assertEqual(symmetries[1], [[1, 0, 1],
                                         [1, 0, 1],
                                         [1, 0, 1]])

        edge_matrix = [[1, 1, 1],
                       [1, 0, 0],
                       [1, 1, 1]]
        symmetries = dnbpy.convert_edge_matrix_to_symmetries(edge_matrix)
        self.assertEqual(len(symmetries), 4)
        self.assertEqual(symmetries[0], [[1, 1, 1],
                                         [1, 0, 0],
                                         [1, 1, 1]])
        self.assertEqual(symmetries[1], [[1, 1, 1],
                                         [0, 0, 1],
                                         [1, 1, 1]])
        self.assertEqual(symmetries[2], [[1, 0, 1],
                                         [1, 0, 1],
                                         [1, 1, 1]])
        self.assertEqual(symmetries[3], [[1, 1, 1],
                                         [1, 0, 1],
                                         [1, 0, 1]])

        edge_matrix = [[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]]
        symmetries = dnbpy.convert_edge_matrix_to_symmetries(edge_matrix)
        self.assertEqual(len(symmetries), 1)
        self.assertEqual(symmetries[0], [[1, 1, 1],
                                         [1, 0, 1],
                                         [1, 1, 1]])

        edge_matrix = [[1, 0, 1],
                       [0, 0, 1],
                       [1, 1, 1]]
        symmetries = dnbpy.convert_edge_matrix_to_symmetries(edge_matrix)
        self.assertEqual(len(symmetries), 4)
        self.assertEqual(symmetries[0], [[1, 0, 1],
                                         [0, 0, 1],
                                         [1, 1, 1]])
        self.assertEqual(symmetries[1], [[1, 0, 1],
                                         [1, 0, 0],
                                         [1, 1, 1]])
        self.assertEqual(symmetries[2], [[1, 1, 1],
                                         [0, 0, 1],
                                         [1, 0, 1]])
        self.assertEqual(symmetries[3], [[1, 1, 1],
                                         [1, 0, 0],
                                         [1, 0, 1]])

    def test_convert_edge_matrix_to_symmetries_1x2(self):
        edge_matrix = [[1, 0, 1, 0, 1],
                       [0, 0, 0, 0, 0],
                       [1, 0, 1, 0, 1]]
        symmetries = dnbpy.convert_edge_matrix_to_symmetries(edge_matrix)
        self.assertEqual(len(symmetries), 1)
        self.assertEqual(symmetries[0], [[1, 0, 1, 0, 1],
                                         [0, 0, 0, 0, 0],
                                         [1, 0, 1, 0, 1]])

        edge_matrix = [[1, 1, 1, 0, 1],
                       [0, 0, 0, 0, 0],
                       [1, 0, 1, 0, 1]]
        symmetries = dnbpy.convert_edge_matrix_to_symmetries(edge_matrix)
        self.assertEqual(len(symmetries), 4)
        self.assertEqual(symmetries[0], [[1, 1, 1, 0, 1],
                                         [0, 0, 0, 0, 0],
                                         [1, 0, 1, 0, 1]])
        self.assertEqual(symmetries[1], [[1, 0, 1, 1, 1],
                                         [0, 0, 0, 0, 0],
                                         [1, 0, 1, 0, 1]])
        self.assertEqual(symmetries[2], [[1, 0, 1, 0, 1],
                                         [0, 0, 0, 0, 0],
                                         [1, 0, 1, 1, 1]])
        self.assertEqual(symmetries[3], [[1, 0, 1, 0, 1],
                                         [0, 0, 0, 0, 0],
                                         [1, 1, 1, 0, 1]])

import unittest

import ai


class TestTabularPolicy(unittest.TestCase):

    def test_create_policy(self):
        policy = ai.TabularPolicy((1, 1), 0.0, 0.0)
        value_table = policy.get_value_table()

        self.assertTrue(len(value_table), 6)
        self.assertTrue('0000' in value_table)
        self.assertTrue('0001' in value_table)
        self.assertTrue('0110' in value_table)
        self.assertTrue('0011' in value_table)
        self.assertTrue('0111' in value_table)
        self.assertTrue('1111' in value_table)

        symmetric_states = policy._symmetric_states
        self.assertTrue(len(symmetric_states), 16)
        self.assertEqual(symmetric_states['0001'], ['0001', '0010', '1000', '0100'])
        self.assertEqual(symmetric_states['0011'], ['0011', '0101', '1010', '1100'])
        self.assertEqual(symmetric_states['1100'], ['1100', '1010', '0101', '0011'])
        self.assertEqual(symmetric_states['1111'], ['1111'])
        self.assertEqual(symmetric_states['1000'], ['1000', '0100', '0001', '0010'])
        self.assertEqual(symmetric_states['1010'], ['1010', '1100', '0101', '0011'])
        self.assertEqual(symmetric_states['0111'], ['0111', '1011', '1110', '1101'])
        self.assertEqual(symmetric_states['1101'], ['1101', '1011', '0111', '1110'])
        self.assertEqual(symmetric_states['0101'], ['0101', '0011', '1010', '1100'])
        self.assertEqual(symmetric_states['1001'], ['1001', '0110'])
        self.assertEqual(symmetric_states['1011'], ['1011', '1101', '1110', '0111'])
        self.assertEqual(symmetric_states['0010'], ['0010', '0100', '1000', '0001'])
        self.assertEqual(symmetric_states['0100'], ['0100', '0010', '0001', '1000'])
        self.assertEqual(symmetric_states['0110'], ['0110', '1001'])
        self.assertEqual(symmetric_states['0000'], ['0000'])
        self.assertEqual(symmetric_states['1110'], ['1110', '1101', '0111', '1011'])

    def test_select_edge(self):
        value_table = {'0000': 0.0, '0010': 0.6, '0110': 0.7, '0011': 0.8, '0111': 0.9, '1111': 0.0}
        policy = ai.TabularPolicy((1, 1), 0.0, 0.0, existing_value_table=value_table)

        edge = policy.select_edge([0, 0, 0, 0])
        self.assertEqual(edge, 0)

        edge = policy.select_edge([1, 0, 0, 0])
        self.assertEqual(edge, 1)

        edge = policy.select_edge([1, 1, 0, 0])
        self.assertEqual(edge, 2)

        edge = policy.select_edge([1, 1, 1, 0])
        self.assertEqual(edge, 3)
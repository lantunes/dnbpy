import unittest

from dnbpy.util.rate_util import *


class TestRateUtil(unittest.TestCase):

    def test_gen_rate_step(self):
        schedule = {0: 0.005, 200: 0.0002, 400: 0.0001}

        rate = gen_rate_step(0, schedule=schedule)
        self.assertEqual(rate, 0.005)

        rate = gen_rate_step(1, schedule=schedule)
        self.assertEqual(rate, 0.005)

        rate = gen_rate_step(200, schedule=schedule)
        self.assertEqual(rate, 0.005)

        rate = gen_rate_step(201, schedule=schedule)
        self.assertEqual(rate, 0.0002)

        rate = gen_rate_step(400, schedule=schedule)
        self.assertEqual(rate, 0.0002)

        rate = gen_rate_step(401, schedule=schedule)
        self.assertEqual(rate, 0.0001)

        rate = gen_rate_step(600, schedule=schedule)
        self.assertEqual(rate, 0.0001)

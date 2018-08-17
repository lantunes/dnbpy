import random
from ai import *


class OpponentPool:
    def __init__(self, max_size):
        self._opponent_pool = []
        self._opponent_pool_max_size = max_size

    def add_to_pool(self, agent):
        self._opponent_pool.append(agent)
        self._opponent_pool = self._opponent_pool[-self._opponent_pool_max_size:]

    def sample_opponent(self, epsilon=0.0):
        if random.random() < epsilon:
            return RandomPolicy()
        else:
            return random.choice(self._opponent_pool)

    def sample_opponent_sep(self, epsilon=0.0):
        if random.random() < epsilon:
            return RandomPolicy()
        else:
            tuple1 = random.choice(self._opponent_pool)
            tuple2 = random.choice(self._opponent_pool)
            return (tuple1[0],tuple2[1])

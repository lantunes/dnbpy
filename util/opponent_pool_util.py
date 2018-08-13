import random
from ai import *


class OpponentPool:
    def __init__(self, max_size):
        self._opponent_pool = []
        self._opponent_pool_max_size = max_size

    def add_to_pool(self, agent):
        self._opponent_pool.append(agent)
        self._opponent_pool = self._opponent_pool[-self._opponent_pool_max_size:]

    def get_opponents(self):
        return list(self._opponent_pool)

    def sample_opponent(self, epsilon=0.0):
        if random.random() < epsilon:
            return RandomPolicy()
        else:
            return random.choice(self._opponent_pool)

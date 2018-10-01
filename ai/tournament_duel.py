from ai import *
import time
from util.initializer_util import *
from util.duel import *

board_size = (3, 3)

pg_params = read_params('../resources/dnbpy38-3x3-relu-351000.txt')
pg_model = PGPolicy3x3CNN(board_size, existing_params=pg_params, activation=tf.nn.relu)
pg_model.set_boltzmann_action(False)
pg_model.set_epsilon(0.0)

pg_params2 = read_params('../resources/dnbpy43-3x3-tanh-305000.txt')
pg_model2 = PGPolicy3x3CNN(board_size, existing_params=pg_params2, activation=tf.nn.tanh)
pg_model2.set_boltzmann_action(False)
pg_model2.set_epsilon(0.0)

player1 = pg_model
player2 = pg_model2

# player2 = MCTSPolicy2(board_size, num_playouts=500, default_policy=pg_model)
# player2 = MCTSPolicyNetPolicy(board_size, num_playouts=1000, w=100, default_policy=pg_model)

results = duel(board_size, player1, player2)

print(results)
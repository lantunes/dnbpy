from ai import *
import time
from util.initializer_util import *
from util.duel import *

board_size = (3, 3)

# pg_params = read_params('../out/dnbpy48/search-weights-433000.txt')
# pg_params = read_params('../out/dnbpy54/weights-244.txt')
pg_params = read_params('../out/dnbpy54/weights-262.txt')
# pg_params = read_params('../out/dnbpy54/weights-309.txt')
# pg_params = read_params('../out/dnbpy54/weights-323.txt')
# pg_params = read_params('../out/dnbpy55/weights-63.txt')
pg_model = PGPolicy3x3CNN(board_size, existing_params=pg_params, activation=tf.nn.relu)
pg_model.set_boltzmann_action(False)
pg_model.set_epsilon(0.0)

pg_params2 = read_params('../out/dnbpy57/weights-139.txt')
pg_model2 = PGPolicy3x3CNN(board_size, existing_params=pg_params2, activation=tf.nn.relu)
pg_model2.set_boltzmann_action(False)
pg_model2.set_epsilon(0.0)

player1 = pg_model
player2 = pg_model2
# player2 = CausalEntropicPolicy(board_size, max_sample_paths=1000)
# player2 = Level2HeuristicPolicy(board_size)

# player1 = MCTSPolicy2(board_size, num_playouts=1000, default_policy=pg_model2)
# player1 = MCTSPolicyNetPolicy(board_size, num_playouts=1000, w=100, default_policy=pg_model)
# player1 = MCTSPolicyNetPolicyCpuct(board_size, num_playouts=1000, cpuct=5, normalize_policy_probs_with_softmax=False)

results = duel(board_size, player1, player2, pg_model)

print(results)

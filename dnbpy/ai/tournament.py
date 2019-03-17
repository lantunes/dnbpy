from dnbpy.ai import *
import time
from dnbpy.util.initializer_util import *

board_size = (3, 3)
n_games = 10000

# player1 = RandomPolicy()
# player1 = Level1HeuristicPolicy(board_size)
player1 = Level2HeuristicPolicy(board_size)
# player1 = MCTSPolicy(board_size, num_playouts=5000, reset_tree=True)
# player1 = MCTSPolicy2(board_size, num_playouts=100000)
# player1 = MCTSPolicy2(board_size, num_playouts=10000, default_policy=Level2HeuristicPolicy(board_size))
# player2 = RandomPolicy()
# player2 = Level1HeuristicPolicy(board_size)
# player2 = Level2HeuristicPolicy(board_size)
# player2 = CausalEntropicPolicy(board_size, max_sample_paths=10000)
# player2 = MCTSPolicy(board_size, num_playouts=5000, reset_tree=True)
# player2 = MCTSPolicy2(board_size, num_playouts=1000)
# player2 = MCTSRootParallelPolicy(board_size, num_playouts=1000, num_workers=8)
# player2 = MCTSPolicy3(board_size, num_playouts=1000)
# player1 = MCTSPolicy2(board_size, num_playouts=100000, default_policy=Level2HeuristicPolicy(board_size))
# player1 = MCTSRootParallelPolicy(board_size, num_playouts=1500, num_workers=35, default_policy=Level2HeuristicPolicy(board_size))

# pg_params = read_params('../resources/pg_2x2_cnn2_tanh_mcts_exit_03-episode-912000.txt')  # current best model
# pg_params = read_params('../resources/pg_2x2_pg_tanh_mcts_exit_probs-episode-658000.txt')
pg_params = read_params('../out/dnbpy53/weights-205.txt')
pg_model = PGPolicy3x3CNN(board_size, existing_params=pg_params, activation=tf.nn.relu)
pg_model.set_boltzmann_action(False)
pg_model.set_epsilon(0.0)

player2 = pg_model
# player2 = MCTSPolicy2(board_size, num_playouts=5000, default_policy=pg_model)
# player2 = MCTSPolicyNetPolicy(board_size, num_playouts=5000, w=1000)

# player2 = MCTSPolicy2(board_size, num_playouts=10000, default_policy=Level2HeuristicPolicy(board_size))
# player2 = MCTSPolicyNetPolicy(board_size, num_playouts=1000, w=10, default_policy=Level2HeuristicPolicy(board_size))
# player2 = MCTSPolicyNetPolicy(board_size, num_playouts=1000, w=10)
# player2 = MCTSPolicyNetPolicyCpuct(board_size, num_playouts=1000, cpuct=5, normalize_policy_probs_with_softmax=False)
# player2 = MCTSPolicyNetPolicyCpuct(board_size, num_playouts=1000, cpuct=5, normalize_policy_probs_with_softmax=False, default_policy=pg_model)

# player2 = MCTSPolicyNetPolicyCpuct(board_size, num_playouts=1000, cpuct=5)
# player2 = MCTSPolicyNetPolicyCpuct(board_size, num_playouts=1000, cpuct=5, default_policy=pg_model)

print("board size: %sx%s" % board_size)
print("player 1 (%s) vs. player 2 (%s)" % (player1.__class__.__name__, player2.__class__.__name__))

p1_results = {'won': 0, 'lost': 0, 'tied': 0}
p2_results = {'won': 0, 'lost': 0, 'tied': 0}
for trial in range(n_games):
    start = time.time()
    players = ['p1', 'p2']
    if trial % 2 == 0:
        players = [x for x in reversed(players)]
    game = Game(board_size, players)
    current_player = game.get_current_player()
    while not game.is_finished():
        board_state = game.get_board_state()
        if current_player == 'p1':
            edge = player1.select_edge(board_state)
            # edge = player1.select_edge(board_state, game.get_score('p1'))  # used for MCTS players
            # edge = player1.select_edge(board_state, game.get_score('p1'), pg_model)  # used for MCTS Policy Net players
            current_player, _ = game.select_edge(edge, 'p1')
        else:
            edge = player2.select_edge(board_state)
            # edge = player2.select_edge(board_state, game.get_score('p2'))  # used for MCTS players
            # edge = player2.select_edge(board_state, game.get_score('p2'), pg_model)  # used for MCTS Policy Net players
            current_player, _ = game.select_edge(edge, 'p2')
    p1_score = game.get_score('p1')
    p2_score = game.get_score('p2')
    if p1_score > p2_score:
        p1_results['won'] += 1
        p2_results['lost'] += 1
    elif p2_score > p1_score:
        p2_results['won'] += 1
        p1_results['lost'] += 1
    else:
        p1_results['tied'] += 1
        p2_results['tied'] += 1
    end = time.time()
    print ("%s: %s; %s: %s (% s)" % (player1.__class__.__name__, p1_results, player2.__class__.__name__, p2_results, (end - start)))
# print ("%s: %s; %s: %s" % (player1.__class__.__name__, p1_results, player2.__class__.__name__, p2_results))

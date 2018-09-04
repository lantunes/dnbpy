from ai import *
from dnbpy import *
from util.helper_functions import *
from util.file_helper import *
from util.reward_util import *
from util.rate_util import *
from util.evaluator import *
from util.duel import *
import concurrent.futures

board_size = (3, 3)
num_episodes = 1000000
learning_rate = 0.005
min_learning_rate = 0.005
epsilon = 0.1
batch_size = 32
decay_speed = 1.0
use_symmetries = True
num_episodes_per_policy_update = 100
num_episodes_per_mcts_policy_duel = 10000
num_games_duel_win_margin = 24
# episodes_per_thread = [100]  # 1 core
# episodes_per_thread = [50, 50]  # 2 cores
# episodes_per_thread = [33, 33, 34]  # 4 cores
# episodes_per_thread = [14, 14, 14, 14, 14, 15, 15]  # 8 cores
# episodes_per_thread = [6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]  # 16 cores
# episodes_per_thread = [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2]  # 36 cores
# episodes_per_thread = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]  # 72 cores
episodes_per_thread = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2] # 50 cores (even)
mcts_simulations = 1000
mcts_w = 100
activation = tf.nn.relu
dropout_keep_prob = 0.5
base_path = get_base_path_arg()

print("initializing for (%s, %s) game..." % (board_size[0], board_size[1]))

pg_params = read_params('resources/dnbpy38-3x3-relu-351000.txt')
curr_policy = PGPolicy3x3CNN(board_size, batch_size=batch_size, dropout_keep_prob=dropout_keep_prob,
                             existing_params=pg_params, activation=activation)
curr_policy.set_boltzmann_action(False)
curr_policy.set_epsilon(0.0)
opp_params = curr_policy.get_params()
mcts_player = MCTSPolicyNetPolicy(board_size, num_playouts=mcts_simulations, w=mcts_w)
reward_fn = DelayedBinaryReward()

print_info(board_size=board_size, num_episodes=num_episodes, policy=curr_policy, mode='MCTS ExIt', reward=reward_fn,
           updates='offline', learning_rate=learning_rate, min_learning_rate=min_learning_rate,
           architecture=curr_policy.get_architecture(), batch_size=batch_size, decay_speed=decay_speed,
           num_episodes_per_policy_update=num_episodes_per_policy_update, episodes_per_thread=episodes_per_thread,
           mcts=mcts_player, mcts_simulations=mcts_simulations, activation=activation, epsilon=epsilon)
print("num episodes per policy_duel: %s" % num_episodes_per_mcts_policy_duel)
print("duel win margin: %s" % num_games_duel_win_margin)
print("MCTS-N w: %s" % mcts_w)

unique_states_visited = set()
all_transitions = []
current_episode_num = 0


def run_episodes(num_episodes_per_thread, existing_params):
    pol = PGPolicy3x3CNN(board_size, batch_size=batch_size, existing_params=existing_params, activation=activation)
    pol.set_boltzmann_action(False)
    pol.set_epsilon(epsilon)

    opp_pol = PGPolicy3x3CNN(board_size, batch_size=batch_size, existing_params=opp_params, activation=activation)
    opp_pol.set_boltzmann_action(False)
    opp_pol.set_epsilon(0.0)
    opp = MCTSPolicyNetPolicy(board_size, num_playouts=mcts_simulations, w=mcts_w)

    run_transitions = []
    for episode_num in range(1, num_episodes_per_thread + 1):
        players = ['policy', 'opponent'] if episode_num % 2 == 0 else ['opponent', 'policy']
        game = Game(board_size, players)
        current_player = game.get_current_player()

        pol_actions = []
        opp_actions = []
        pol_states = []
        opp_states = []

        while not game.is_finished():
            board_state = game.get_board_state()
            if current_player == 'policy':
                pol_states.append(board_state)
                edge = pol.select_edge(board_state)
                pol_actions.append(to_one_hot_action(board_state, edge))
                current_player, _ = game.select_edge(edge, current_player)
                unique_states_visited.add(as_string(game.get_board_state()))
            else:
                opp_states.append(board_state)
                edge = opp.select_edge(board_state, game.get_score(current_player), opp_pol)
                opp_actions.append(to_one_hot_action(board_state, edge))
                current_player, _ = game.select_edge(edge, current_player)
                unique_states_visited.add(as_string(game.get_board_state()))

        pol_reward = reward_fn.compute_reward(game, 'policy', 'opponent')
        opp_reward = reward_fn.compute_reward(game, 'opponent', 'policy')

        # don't add transitions that have 0 reward as the gradient will be zero anyways
        if pol_reward == 1:
            pol_outcomes = len(pol_actions)*[pol_reward]
            append_transitions(pol_states, pol_actions, pol_outcomes, run_transitions, use_symmetries, board_size)
        elif opp_reward == 1:
            opp_outcomes = len(opp_actions)*[opp_reward]
            append_transitions(opp_states, opp_actions, opp_outcomes, run_transitions, use_symmetries, board_size)
    return run_transitions


def have_duel(existing_params):
    pol = PGPolicy3x3CNN(board_size, batch_size=batch_size, existing_params=existing_params, activation=activation)
    pol.set_boltzmann_action(False)
    pol.set_epsilon(0.0)

    mcts_pol = PGPolicy3x3CNN(board_size, batch_size=batch_size, existing_params=opp_params, activation=activation)
    mcts_pol.set_boltzmann_action(False)
    mcts_pol.set_epsilon(0.0)

    duel_results = duel(board_size, pol, mcts_pol)
    return duel_results


while current_episode_num < num_episodes:
    existing_params = curr_policy.get_params()

    if current_episode_num % num_episodes_per_mcts_policy_duel == 0:
        # have a duel to see if we should update the MCTS-N policy (on separate process, so as to avoid hanging bug)
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            for r in executor.map(have_duel, [existing_params]):
                duel_results = r
        won = int(duel_results['won'])
        lost = int(duel_results['lost'])
        if (won - lost) >= num_games_duel_win_margin:
            print("current policy beats previous policy: %s" % duel_results)
            opp_params = existing_params

    with concurrent.futures.ProcessPoolExecutor(max_workers=len(episodes_per_thread)) as executor:
        for r in executor.map(run_episodes, episodes_per_thread, [existing_params]*len(episodes_per_thread)):
            all_transitions.extend(r)
    current_episode_num += sum(episodes_per_thread)

    lr = gen_rate_exponential(current_episode_num, learning_rate, min_learning_rate, num_episodes, decay_speed)
    curr_policy.set_learning_rate(lr)

    if current_episode_num % num_episodes_per_policy_update == 0:
        curr_policy.update_model(all_transitions)
        all_transitions = []

    # analyze results
    if current_episode_num % 1000 == 0:
        # play against opponents
        curr_policy.set_boltzmann_action(False)
        curr_policy.set_epsilon(0.0)
        opponents = [RandomPolicy(), Level1HeuristicPolicy(board_size), Level2HeuristicPolicy(board_size)]
        results = evaluate(curr_policy, board_size, 1000, opponents)
        print("%s, %s, %s, %s, %s, %s, %s, %s" % (current_episode_num, results[RandomPolicy.__name__]['won'],
                                                  results[Level1HeuristicPolicy.__name__]['won'],
                                                  results[Level2HeuristicPolicy.__name__]['won'],
                                                  results, len(unique_states_visited), epsilon, lr))
        WeightWriter.print_episode(base_path, current_episode_num, curr_policy.print_params)

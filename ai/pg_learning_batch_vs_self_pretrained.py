from ai import *
from dnbpy import *
from util.helper_functions import *
from util.file_helper import *
from util.reward_util import *
from util.rate_util import *
from util.evaluator import *
from util.opponent_pool_util import *

board_size = (2, 2)
num_episodes = 1000000
learning_rate_schedule = {0: 0.005, 700000: 0.0005}
epsilon_schedule = {0: 0.4, 200000: 0.3}
batch_size = 32
decay_speed = 1.0
opponent_pool_max_size = 100
num_episodes_per_policy_update = 500
num_episodes_per_opponent_cache = 500
dropout_keep_prob = 0.5
use_symmetries = True
base_path = get_base_path_arg()

print("initializing for (%s, %s) game..." % (board_size[0], board_size[1]))

pg_params = read_params('../resources/pg_2x2_cnn2_vs_L0_L1_L2_batch_01-episode-1496000.txt')
policy = PGPolicyCNN2(board_size, batch_size=batch_size, dropout_keep_prob=dropout_keep_prob, existing_params=pg_params)
opponent = policy
reward_fn = DelayedBinaryReward()
opponent_pool = OpponentPool(max_size=opponent_pool_max_size)

print_info(board_size=board_size, num_episodes=num_episodes, policy=policy, mode='self-play pool', reward=reward_fn,
           updates='offline', learning_rate_schedule=learning_rate_schedule, epsilon_schedule=epsilon_schedule,
           architecture=policy.get_architecture(), batch_size=batch_size, decay_speed=decay_speed)

unique_states_visited = set()
all_transitions = []
for episode_num in range(1, num_episodes + 1):
    eps = gen_rate_step(episode_num, epsilon_schedule)
    lr = gen_rate_step(episode_num, learning_rate_schedule)
    policy.set_boltzmann_action(False)
    policy.set_epsilon(eps)
    policy.set_learning_rate(lr)

    policy_actions = []
    opponent_actions = []
    policy_states = []
    opponent_states = []

    players = ['policy', 'opponent'] if episode_num % 2 == 0 else ['opponent', 'policy']
    game = Game(board_size, players)
    current_player = game.get_current_player()
    while not game.is_finished():
        board_state = game.get_board_state()
        if current_player == 'policy':
            policy_states.append(board_state)
            edge = policy.select_edge(board_state)
            policy_actions.append(to_one_hot_action(board_state, edge))
            current_player, _ = game.select_edge(edge, current_player)
            unique_states_visited.add(as_string(game.get_board_state()))
        else:
            opponent_states.append(board_state)
            edge = opponent.select_edge(board_state)
            opponent_actions.append(to_one_hot_action(board_state, edge))
            current_player, _ = game.select_edge(edge, current_player)
            unique_states_visited.add(as_string(game.get_board_state()))
    policy_reward = reward_fn.compute_reward(game, 'policy', 'opponent')
    opponent_reward = reward_fn.compute_reward(game, 'opponent', 'policy')

    # don't add transitions that have 0 reward as the gradient will be zero anyways
    if policy_reward == 1:
        policy_outcomes = len(policy_actions)*[policy_reward]
        append_transitions(policy_states, policy_actions, policy_outcomes, all_transitions, use_symmetries, board_size)
    elif opponent_reward == 1:
        opponent_outcomes = len(opponent_actions)*[opponent_reward]
        append_transitions(opponent_states, opponent_actions, opponent_outcomes, all_transitions, use_symmetries, board_size)

    if episode_num % num_episodes_per_policy_update == 0:
        policy.update_model(all_transitions)
        all_transitions = []

    if episode_num % num_episodes_per_opponent_cache == 0:
        opponent_pool.add_to_pool(policy.copy())
        opponent = opponent_pool.sample_opponent()

    # analyze results
    if episode_num % 1000 == 0:
        # play against opponents
        policy.set_boltzmann_action(False)
        policy.set_epsilon(0.0)
        opponents = [RandomPolicy(), Level1HeuristicPolicy(board_size), Level2HeuristicPolicy(board_size)]
        results = evaluate(policy, board_size, 1000, opponents)
        print("%s, %s, %s, %s, %s, %s, %s, %s" % (episode_num, results[RandomPolicy.__name__]['won'],
                                                  results[Level1HeuristicPolicy.__name__]['won'],
                                                  results[Level2HeuristicPolicy.__name__]['won'],
                                                  results, len(unique_states_visited), eps, lr))
        WeightWriter.print_episode(base_path, episode_num, policy.print_params)
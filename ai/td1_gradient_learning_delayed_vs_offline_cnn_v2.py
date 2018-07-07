from ai import *
from dnbpy import *
from util.helper_functions import *
from util.file_helper import *
from util.reward_util import *
from util.rate_util import *
from util.evaluator import *

board_size = (2, 2)
num_episodes = 300000
learning_rate = 0.005
min_learning_rate = 0.000001
epsilon = 1.0
min_epsilon = 0.01
decay_speed = 1.0
base_path = get_base_path_arg()
rand_prob = 0.5

print("initializing for (%s, %s) game..." % (board_size[0], board_size[1]))

policy = TDOneGradientPolicyCNNV2b(board_size=board_size)
opponent = RandomPolicy()
reward_fn = DelayedBinaryReward()

print_info(board_size=board_size, num_episodes=num_episodes, policy=policy, mode='self-play and vs. Random', reward=reward_fn,
           updates='offline', learning_rate=learning_rate, min_learning_rate=min_learning_rate, epsilon=epsilon,
           min_epsilon=min_epsilon, architecture=policy.get_architecture(), decay_speed=decay_speed, rand_prob=rand_prob)


unique_states_visited = set()
games_played_against_random = 0
for episode_num in range(1, num_episodes + 1):
    eps = gen_rate_exponential(episode_num, epsilon, min_epsilon, num_episodes, decay_speed)
    lr = gen_rate_exponential(episode_num, learning_rate, min_learning_rate, num_episodes, decay_speed)
    policy.set_epsilon(eps)
    policy.set_learning_rate(lr)
    policy.reset_history_buffer()

    if random.random() < rand_prob:
        players = ['policy', 'opponent'] if games_played_against_random % 2 == 0 else ['opponent', 'policy']
        game = Game(board_size, players)
        current_player = game.get_current_player()
        prediction_history = []
        prediction_gradient_history = []
        while not game.is_finished():
            board_state = game.get_board_state()
            if current_player == 'policy':
                edge = policy.select_edge(board_state)
                current_player, _ = game.select_edge(edge, 'policy')
                prediction_history.append(policy.get_last_prediction())
                prediction_gradient_history.append(policy.get_last_prediction_gradient())
                unique_states_visited.add(as_string(game.get_board_state()))
            else:
                edge = opponent.select_edge(board_state)
                current_player, _ = game.select_edge(edge, 'opponent')
        reward = reward_fn.compute_reward(game, 'policy', 'opponent')
        policy.update_offline(prediction_history, prediction_gradient_history, reward)
        games_played_against_random += 1
    else:
        players = [0, 1]
        game = Game(board_size, players)
        current_player = game.get_current_player()
        prediction_history_p1 = []
        prediction_gradient_history_p1 = []
        prediction_history_p2 = []
        prediction_gradient_history_p2 = []
        while not game.is_finished():
            board_state = game.get_board_state()
            if current_player == 0:
                edge = policy.select_edge(board_state)
                current_player, _ = game.select_edge(edge, 0)
                prediction_history_p1.append(policy.get_last_prediction())
                prediction_gradient_history_p1.append(policy.get_last_prediction_gradient())
                unique_states_visited.add(as_string(game.get_board_state()))
            else:
                edge = policy.select_edge(board_state)
                current_player, _ = game.select_edge(edge, 1)
                prediction_history_p2.append(policy.get_last_prediction())
                prediction_gradient_history_p2.append(policy.get_last_prediction_gradient())
                unique_states_visited.add(as_string(game.get_board_state()))
        reward = reward_fn.compute_reward(game, 0, 1)
        policy.update_offline(prediction_history_p1, prediction_gradient_history_p1, reward)
        reward = reward_fn.compute_reward(game, 1, 0)
        policy.update_offline(prediction_history_p2, prediction_gradient_history_p2, reward)

    # analyze results
    if episode_num % 500 == 0:
        # play against opponents
        policy.set_epsilon(0.0)
        opponents = [RandomPolicy(), Level1HeuristicPolicy(board_size), Level2HeuristicPolicy(board_size)]
        results = evaluate(policy, board_size, 500, opponents)
        print("%s, %s, %s, %s, %s, %s, %s, %s" % (episode_num, results[opponents[0].__class__.__name__]['won'],
                                                  results[opponents[1].__class__.__name__]['won'],
                                                  results[opponents[2].__class__.__name__]['won'],
                                                  results, len(unique_states_visited), eps, lr))
        WeightWriter.print_episode(base_path, episode_num, policy.print_params)

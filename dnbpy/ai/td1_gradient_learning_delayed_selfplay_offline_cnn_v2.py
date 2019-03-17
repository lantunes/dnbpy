from dnbpy.ai import *
from dnbpy.util.helper_functions import *
from dnbpy.util.file_helper import *
from dnbpy.util.reward_util import *
from dnbpy.util.rate_util import *

board_size = (2, 2)
num_episodes = 300000
learning_rate = 0.005
min_learning_rate = 0.000001
epsilon = 1.0
min_epsilon = 0.01
decay_speed = 1.0
base_path = get_base_path_arg()

print("initializing for (%s, %s) game..." % (board_size[0], board_size[1]))

policy = TDOneGradientPolicyCNNV2(board_size=board_size)
random_policy = RandomPolicy()
reward_fn = DelayedBinaryReward()

print_info(board_size=board_size, num_episodes=num_episodes, policy=policy, mode='self-play', reward=reward_fn,
           updates='offline', learning_rate=learning_rate, min_learning_rate=min_learning_rate, epsilon=epsilon,
           min_epsilon=min_epsilon, architecture=policy.get_architecture(), decay_speed=decay_speed)


unique_states_visited = set()
for episode_num in range(1, num_episodes + 1):
    eps = gen_rate_exponential(episode_num, epsilon, min_epsilon, num_episodes, decay_speed)
    lr = gen_rate_exponential(episode_num, learning_rate, min_learning_rate, num_episodes, decay_speed)
    policy.set_epsilon(eps)
    policy.set_learning_rate(lr)
    policy.reset_history_buffer()
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
        # play against random opponent
        policy.set_epsilon(0.0)
        results = {'won': 0, 'lost': 0, 'tied': 0}
        for trial in range(500):
            players = ['policy', 'random']
            if trial % 2 == 0:
                players = [x for x in reversed(players)]
            game = Game(board_size, players)
            current_player = game.get_current_player()
            while not game.is_finished():
                board_state = game.get_board_state()
                if current_player == 'random':
                    edge = random_policy.select_edge(board_state)
                    current_player, _ = game.select_edge(edge, 'random')
                else:
                    edge = policy.select_edge(board_state)
                    current_player, _ = game.select_edge(edge, 'policy')
            policy_score = game.get_score('policy')
            random_score = game.get_score('random')
            if policy_score > random_score:
                results['won'] += 1
            elif random_score > policy_score:
                results['lost'] += 1
            else:
                results['tied'] += 1
        print("%s, %s, %s, %s (%s, %s)" % (episode_num, results['won'], results,
                                           len(unique_states_visited), eps, lr))
        WeightWriter.print_episode(base_path, episode_num, policy.print_params)

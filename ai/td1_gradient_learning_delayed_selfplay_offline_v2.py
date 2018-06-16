from ai import *
from dnbpy import *
from util.helper_functions import *

board_size = (2, 2)
num_episodes = 200000
learning_rate = 0.05
min_learning_rate = 0.0001
epsilon = 0.99
min_epsilon = 0.05

print("initializing for (%s, %s) game..." % (board_size[0], board_size[1]))

policy = TDOneGradientPolicyV2(board_size=board_size)
random_policy = RandomPolicy()

print_info(board_size=board_size, num_episodes=num_episodes, policy=policy, mode='self-play', reward='delayed',
           updates='offline', learning_rate=learning_rate, min_learning_rate=min_learning_rate, epsilon=epsilon,
           min_epsilon=min_epsilon, architecture=policy.get_architecture())


def gen_rate(iteration,l_max,l_min,N_max):
    if iteration>N_max: return (l_min)
    alpha = 2*l_max
    beta = np.log((alpha/l_min-1))/N_max
    return (alpha/(1+np.exp(beta*iteration)))


def compute_reward(game, player_to_update):
    if game.is_finished() and game.get_score(player_to_update) > game.get_score(abs(1 - player_to_update)):
        return 1.0
    return 0.0

unique_states_visited = set()
for episode_num in range(1, num_episodes + 1):
    eps = gen_rate(episode_num, epsilon, min_epsilon, num_episodes)
    lr = gen_rate(episode_num, learning_rate, min_learning_rate, num_episodes)
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

    reward = compute_reward(game, 0)
    policy.update_offline(prediction_history_p1, prediction_gradient_history_p1, reward)
    reward = compute_reward(game, 1)
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
policy.print_params()

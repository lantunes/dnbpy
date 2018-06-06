from ai import *
from dnbpy import *
from util.state_space_metrics import *
from util.helper_functions import *

board_size = (2, 2)
num_episodes = 300000
learning_rate = 1.0
min_learning_rate = 1e-6
gamma = 0.99
epsilon = 0.99
min_epsilon = 0.99

print("initializing value table for (%s, %s) game..." % (board_size[0], board_size[1]))

policy = TDOnePolicy(board_size=board_size, epsilon=epsilon, learning_rate=learning_rate, gamma=gamma,
                 initial_state_value=lambda: random.random())
random_policy = RandomPolicy()

print_info(board_size=board_size, num_episodes=num_episodes, policy=policy, mode='self-play', reward='delayed',
           updates='offline', learning_rate=learning_rate, min_learning_rate=min_learning_rate, epsilon=epsilon,
           min_epsilon=min_epsilon)


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
state_space_metrics = StateSpaceMetrics(board_size)
for episode_num in range(1, num_episodes + 1):
    eps = gen_rate(episode_num, epsilon, min_epsilon, num_episodes)
    lr = gen_rate(episode_num, learning_rate, min_learning_rate, num_episodes)
    policy.set_epsilon(eps)
    policy.set_learning_rate(lr)
    players = [0, 1]
    # player_to_update = 0 if episode_num % 2 == 0 else 1
    game = Game(board_size, players)
    current_player = game.get_current_player()
    backups_p1 = []
    backups_p2 = []
    while not game.is_finished():
        board_state = game.get_board_state()
        if current_player == 0:
            edge = policy.select_edge(board_state)
            current_player, _ = game.select_edge(edge, 0)
            if not game.is_finished():
                backups_p1.append(game.get_board_state())
            unique_states_visited.add(as_string(game.get_board_state()))
            state_space_metrics.state_visited(as_string(game.get_board_state()))
        else:
            edge = policy.select_edge(board_state)
            current_player, _ = game.select_edge(edge, 1)
            if not game.is_finished():
                backups_p2.append(game.get_board_state())
            unique_states_visited.add(as_string(game.get_board_state()))
            state_space_metrics.state_visited(as_string(game.get_board_state()))

    reward = compute_reward(game, 0)
    policy.update_value(reward, backups_p1)
    reward = compute_reward(game, 1)
    policy.update_value(reward, backups_p2)
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
        vt = policy.get_value_table()
        print("%s, %s, %s, %s, %s (%s, %s)" % (episode_num, results['won'], results, len([x for x in vt if vt[x] > 0.0 ]),
                                      len(unique_states_visited), eps, lr))

print(policy.get_value_table())
print(state_space_metrics.get_state_visitation_counts())
state_space_metrics.plot_state_visitation_count_heatmap([64, 64])

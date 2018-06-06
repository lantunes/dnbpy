from ai import *
from dnbpy import *
from util.helper_functions import *

board_size = (2, 2)
num_episodes = 200000
learning_rate = 0.05
min_learning_rate = 0.0001
temperature = 1.0
min_temperature = 0.001

print("initializing for (%s, %s) game..." % (board_size[0], board_size[1]))

policy = TDOneGradientPolicy(board_size=board_size)
policy.set_epsilon(0.0)
random_policy = RandomPolicy()

print_info(board_size=board_size, num_episodes=num_episodes, policy=policy, mode='self-play', reward='delayed',
           updates='online', learning_rate=learning_rate, min_learning_rate=min_learning_rate, temperature=temperature,
           min_temperature=min_temperature)


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
    tmp = gen_rate(episode_num, temperature, min_temperature, num_episodes)
    lr = gen_rate(episode_num, learning_rate, min_learning_rate, num_episodes)
    policy.set_softmax_action(True)
    policy.set_temperature(tmp)
    policy.set_learning_rate(lr)
    policy.reset_history()
    players = [0, 1]
    player_to_update = 0 if episode_num % 2 == 0 else 1
    game = Game(board_size, players)
    current_player = game.get_current_player()
    while not game.is_finished():
        board_state = game.get_board_state()
        if current_player == 0:
            policy.set_should_store_history(player_to_update == 0)
            edge = policy.select_edge(board_state)
            current_player, _ = game.select_edge(edge, 0)
            if not game.is_finished() and player_to_update == 0:
                policy.update()
            if player_to_update == 0: unique_states_visited.add(as_string(game.get_board_state()))
        else:
            policy.set_should_store_history(player_to_update == 1)
            edge = policy.select_edge(board_state)
            current_player, _ = game.select_edge(edge, 1)
            if not game.is_finished() and player_to_update == 1:
                policy.update()
            if player_to_update == 1: unique_states_visited.add(as_string(game.get_board_state()))

    reward = compute_reward(game, player_to_update)
    policy.update_terminal(reward)
    # analyze results
    if episode_num % 500 == 0:
        # policy.print_gradients()
        # play against random opponent
        policy.set_softmax_action(False)
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
                                           len(unique_states_visited), tmp, lr))


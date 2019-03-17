from dnbpy.ai import *
from dnbpy.util.helper_functions import *
from dnbpy.util.file_helper import *
from dnbpy.util.reward_util import *
from dnbpy.util.rate_util import *

board_size = (2, 2)
num_episodes = 100000
learning_rate = 0.5
min_learning_rate = 0.05
epsilon = 0.99
min_epsilon = 0.01
decay_speed = 1.0
base_path = get_base_path_arg()

print("initializing for (%s, %s) game..." % (board_size[0], board_size[1]))

policy = TDOneGradientPolicy(board_size=board_size)
policy.set_softmax_action(False)
opponent = TDOneTabularPolicy(board_size=board_size, table_file_path='../resources/td1_2x2_0.6_1.0_0.99_delayed_selfplay100k.txt')
opponent.set_epsilon(0.35)
random_policy = RandomPolicy()
reward_fn = DelayedBinaryReward()

print_info(board_size=board_size, num_episodes=num_episodes, policy=policy, mode='vs. TD1 tabular', reward=reward_fn,
           updates='online', learning_rate=learning_rate, min_learning_rate=min_learning_rate, epsilon=epsilon,
           min_epsilon=min_epsilon, architecture=policy.get_architecture(), decay_speed=decay_speed)


unique_states_visited = set()
for episode_num in range(1, num_episodes + 1):
    eps = gen_rate_exponential(episode_num, epsilon, min_epsilon, num_episodes, decay_speed)
    lr = gen_rate_exponential(episode_num, learning_rate, min_learning_rate, num_episodes, decay_speed)
    policy.set_epsilon(eps)
    policy.set_learning_rate(lr)
    policy.reset_history()
    players = ['policy', 'opponent'] if episode_num % 2 == 0 else ['opponent', 'policy']
    game = Game(board_size, players)
    current_player = game.get_current_player()
    while not game.is_finished():
        board_state = game.get_board_state()
        if current_player == 'policy':
            policy.set_should_store_history(True)
            edge = policy.select_edge(board_state)
            current_player, _ = game.select_edge(edge, 'policy')
            if not game.is_finished():
                policy.update()
            unique_states_visited.add(as_string(game.get_board_state()))
        else:
            edge = opponent.select_edge(board_state)
            current_player, _ = game.select_edge(edge, 'opponent')

    reward = reward_fn.compute_reward(game, 'policy', 'opponent')
    policy.update_terminal(reward)
    # analyze results
    if episode_num % 500 == 0:
        # policy.print_gradients()
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

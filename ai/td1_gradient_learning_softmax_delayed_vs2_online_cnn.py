from ai import *
from dnbpy import *
from util.helper_functions import *
from util.file_helper import *
from util.reward_util import *
from util.rate_util import *
from util.opponent_pool_util import *

board_size = (2, 2)
num_episodes = 100000
learning_rate = 0.005
min_learning_rate = 0.0001
temperature = 1.0
min_temperature = 0.01
decay_speed = 1.0
opponent_pool_max_size = 100
base_path = get_base_path_arg()

print("initializing for (%s, %s) game..." % (board_size[0], board_size[1]))

policy = TDOneGradientPolicyCNN(board_size=board_size)
policy.set_epsilon(0.0)
random_policy = RandomPolicy()
reward_fn = DelayedBinaryReward()
opponent_pool = OpponentPool(max_size=opponent_pool_max_size)
opponent_pool.add_to_pool(RandomPolicy())

print_info(board_size=board_size, num_episodes=num_episodes, policy=policy, mode='vs. Self pool', reward=reward_fn,
           updates='online', learning_rate=learning_rate, min_learning_rate=min_learning_rate, temperature=temperature,
           min_temperature=min_temperature, architecture=policy.get_architecture(), decay_speed=decay_speed)


unique_states_visited = set()
for episode_num in range(1, num_episodes + 1):
    tmp = gen_rate_exponential(episode_num, temperature, min_temperature, num_episodes, decay_speed)
    lr = gen_rate_exponential(episode_num, learning_rate, min_learning_rate, num_episodes, decay_speed)
    policy.set_softmax_action(True)
    policy.set_temperature(tmp)
    policy.set_learning_rate(lr)
    policy.reset_history()
    opponent = opponent_pool.sample_opponent(epsilon=0.5)
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
    policy.set_softmax_action(False)
    opponent_pool.add_to_pool(policy.copy())
    # analyze results
    if episode_num % 500 == 0:
        # play against random opponent
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
        WeightWriter.print_episode(base_path, episode_num, policy.print_params)

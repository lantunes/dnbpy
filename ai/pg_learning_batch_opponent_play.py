from ai import *
from dnbpy import *
from util.helper_functions import *
from util.file_helper import *
from util.reward_util import *
from util.rate_util import *
from util.evaluator import *

import tensorflow as tf

board_size = (2, 2)
num_episodes = 1000000
learning_rate = 0.005
min_learning_rate = 0.000001
temperature = 1.0
min_temperature = 0.01
decay_speed = 1.0
base_path = get_base_path_arg()

print("initializing for (%s, %s) game..." % (board_size[0], board_size[1]))

policy = PGPolicyCNN(board_size, batch_size=32, reduction=tf.reduce_mean)
p0 = Level2HeuristicPolicy(board_size)
p1 = RandomPolicy()
reward_fn = DelayedBinaryReward()

print_info(board_size=board_size, num_episodes=num_episodes, policy=policy, mode='L2 vs L0', reward=reward_fn,
           updates='offline', learning_rate=learning_rate, min_learning_rate=min_learning_rate, temperature=temperature,
           min_temperature=min_temperature, architecture=policy.get_architecture(), decay_speed=decay_speed)


def to_one_hot_action(board_state, edge_index):
    action_vector = [0]*len(board_state)
    action_vector[edge_index] = 1
    return action_vector


def append_transitions(states, actions, outcomes, all_transitions):
    for i, _ in enumerate(actions):
        state = states[i]
        action = actions[i]
        reward = outcomes[i]
        all_transitions.append([state, action, reward])

unique_states_visited = set()
all_transitions = []
for episode_num in range(1, num_episodes + 1):
    lr = gen_rate_exponential(episode_num, learning_rate, min_learning_rate, num_episodes, decay_speed)
    policy.set_learning_rate(lr)
    players = [0, 1] if episode_num % 2 == 0 else [1, 0]
    game = Game(board_size, players)
    current_player = game.get_current_player()

    p0_actions = []
    p1_actions = []
    p0_states = []
    p1_states = []

    while not game.is_finished():
        board_state = game.get_board_state()
        if current_player == 0:
            p0_states.append(board_state)
            edge = p0.select_edge(board_state)
            p0_actions.append(to_one_hot_action(board_state, edge))
            current_player, _ = game.select_edge(edge, 0)
            unique_states_visited.add(as_string(game.get_board_state()))
        else:
            p1_states.append(board_state)
            edge = p1.select_edge(board_state)
            p1_actions.append(to_one_hot_action(board_state, edge))
            current_player, _ = game.select_edge(edge, 1)
            unique_states_visited.add(as_string(game.get_board_state()))

    p0_reward = reward_fn.compute_reward(game, 0, 1)
    p1_reward = reward_fn.compute_reward(game, 1, 0)

    # don't add transitions that have 0 reward as the gradient will be zero anyways
    if p0_reward == 1:
        p0_outcomes = len(p0_actions)*[p0_reward]
        append_transitions(p0_states, p0_actions, p0_outcomes, all_transitions)
    elif p1_reward == 1:
        p1_outcomes = len(p1_actions)*[p1_reward]
        append_transitions(p1_states, p1_actions, p1_outcomes, all_transitions)

    if episode_num % 100 == 0:
        policy.update_model(all_transitions)
        all_transitions = []

    # analyze results
    if episode_num % 1000 == 0:
        # play against opponents
        policy.set_boltzmann_action(False)
        opponents = [RandomPolicy(), Level1HeuristicPolicy(board_size), Level2HeuristicPolicy(board_size)]
        results = evaluate(policy, board_size, 1000, opponents)
        print("%s, %s, %s, %s, %s, %s, %s, %s" % (episode_num, results[RandomPolicy.__name__]['won'],
                                                  results[Level1HeuristicPolicy.__name__]['won'],
                                                  results[Level2HeuristicPolicy.__name__]['won'],
                                                  results, len(unique_states_visited), 1.0, lr))
        WeightWriter.print_episode(base_path, episode_num, policy.print_params)
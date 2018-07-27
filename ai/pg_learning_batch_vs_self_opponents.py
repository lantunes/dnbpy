from ai import *
from dnbpy import *
from util.helper_functions import *
from util.file_helper import *
from util.reward_util import *
from util.rate_util import *
from util.evaluator import *

board_size = (2, 2)
num_episodes = 1000000
learning_rate = 0.005
min_learning_rate = 0.005
epsilon = 0.1
batch_size = 32
decay_speed = 1.0
# play_prob_schedule = {0: 1.0, 300000: 0.0}
play_prob_max = 1.0
play_prob_min = 0.1
use_symmetries = True
base_path = get_base_path_arg()

print("initializing for (%s, %s) game..." % (board_size[0], board_size[1]))

policy = PGPolicyCNN2(board_size, batch_size=batch_size)
# p0 = Level2HeuristicPolicy(board_size)
p0 = MCTSRootParallelPolicy(board_size, num_playouts=250, num_workers=4, default_policy=Level2HeuristicPolicy(board_size))
# p1 = RandomPolicy()
reward_fn = DelayedBinaryReward()

print_info(board_size=board_size, num_episodes=num_episodes, policy=policy, mode='watch opponents or self-play',
           reward=reward_fn, updates='offline', learning_rate=learning_rate, min_learning_rate=min_learning_rate,
           epsilon=epsilon, architecture=policy.get_architecture(), batch_size=batch_size)

unique_states_visited = set()
all_transitions = []
games_played_against_opponents = 0

for episode_num in range(1, num_episodes + 1):
    lr = gen_rate_exponential(episode_num, learning_rate, min_learning_rate, num_episodes, decay_speed)
    # play_prob = gen_rate_step(episode_num, play_prob_schedule)
    play_prob = gen_rate_exponential(episode_num, play_prob_max, play_prob_min, num_episodes, decay_speed)
    policy.set_boltzmann_action(False)
    policy.set_epsilon(epsilon)
    policy.set_learning_rate(lr)

    if random.random() < play_prob:
        p0_actions = []
        p1_actions = []
        p0_states = []
        p1_states = []
        players = [0, 1] if games_played_against_opponents % 2 == 0 else [1, 0]
        game = Game(board_size, players)
        current_player = game.get_current_player()
        while not game.is_finished():
            board_state = game.get_board_state()
            if current_player == 0:
                p0_states.append(board_state)
                edge = p0.select_edge(board_state, game.get_score(current_player))
                p0_actions.append(to_one_hot_action(board_state, edge))
                current_player, _ = game.select_edge(edge, 0)
                unique_states_visited.add(as_string(game.get_board_state()))
            else:
                p1_states.append(board_state)
                edge = p0.select_edge(board_state, game.get_score(current_player))
                p1_actions.append(to_one_hot_action(board_state, edge))
                current_player, _ = game.select_edge(edge, 1)
                unique_states_visited.add(as_string(game.get_board_state()))
        games_played_against_opponents += 1
        p0_reward = reward_fn.compute_reward(game, 0, 1)
        p1_reward = reward_fn.compute_reward(game, 1, 0)
        if p0_reward == 1:
            p0_outcomes = len(p0_actions)*[p0_reward]
            append_transitions(p0_states, p0_actions, p0_outcomes, all_transitions, use_symmetries, board_size)
        elif p1_reward == 1:
            p1_outcomes = len(p1_actions)*[p1_reward]
            append_transitions(p1_states, p1_actions, p1_outcomes, all_transitions, use_symmetries, board_size)
    else:
        policy_actions = []
        opponent_actions = []
        policy_states = []
        opponent_states = []
        players = [0, 1]
        game = Game(board_size, players)
        current_player = game.get_current_player()
        while not game.is_finished():
            board_state = game.get_board_state()
            if current_player == 0:
                policy_states.append(board_state)
                edge = policy.select_edge(board_state)
                policy_actions.append(to_one_hot_action(board_state, edge))
                current_player, _ = game.select_edge(edge, current_player)
                unique_states_visited.add(as_string(game.get_board_state()))
            else:
                opponent_states.append(board_state)
                edge = policy.select_edge(board_state)
                opponent_actions.append(to_one_hot_action(board_state, edge))
                current_player, _ = game.select_edge(edge, current_player)
                unique_states_visited.add(as_string(game.get_board_state()))
        policy_reward = reward_fn.compute_reward(game, 0, 1)
        opponent_reward = reward_fn.compute_reward(game, 1, 0)
        if policy_reward == 1:
            policy_outcomes = len(policy_actions)*[policy_reward]
            append_transitions(policy_states, policy_actions, policy_outcomes, all_transitions)
        elif opponent_reward == 1:
            opponent_outcomes = len(opponent_actions)*[opponent_reward]
            append_transitions(opponent_states, opponent_actions, opponent_outcomes, all_transitions)

    if episode_num % 100 == 0:
        policy.update_model(all_transitions)
        all_transitions = []

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
                                                  results, len(unique_states_visited), play_prob, lr))
        WeightWriter.print_episode(base_path, episode_num, policy.print_params)

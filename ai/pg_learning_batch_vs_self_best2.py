from ai import *
from dnbpy import *
from util.helper_functions import *
from util.file_helper import *
from util.reward_util import *
from util.rate_util import *
from util.evaluator import *
from util.duel import *

board_size = (3, 3)
num_episodes = 1000000
learning_rate_schedule = {0: 0.005, 700000: 0.0005}
policy_epsilon_start = 1.0
policy_epsilon_end = 0.15
best_epsilon = 0.15
batch_size = 32
num_episodes_per_policy_update = 100
num_episodes_per_policy_evaluation = 5000
decay_speed = 1.0
use_symmetries = True
base_path = get_base_path_arg()

print("initializing for (%s, %s) game..." % (board_size[0], board_size[1]))

policy = PGPolicy3x3CNN(board_size, batch_size=batch_size, dropout_keep_prob=0.5)
policy.set_boltzmann_action(False)
reward_fn = DelayedBinaryReward()
best = policy.copy()

print_info(board_size=board_size, num_episodes=num_episodes, policy=policy, mode='self-play pool', reward=reward_fn,
           updates='offline', learning_rate_schedule=learning_rate_schedule, architecture=policy.get_architecture(), 
           batch_size=batch_size, epsilon=policy_epsilon_start, min_epsilon=policy_epsilon_end, decay_speed=decay_speed)


def compute_policy_improvement(duel_result, policy_eval_results, best_eval_results):
    duel_diff = duel_result['won'] - duel_result['lost']
    L0_diff = policy_eval_results[RandomPolicy.__name__]['won'] - best_eval_results[RandomPolicy.__name__]['won']
    L1_diff = policy_eval_results[Level1HeuristicPolicy.__name__]['won'] - best_eval_results[Level1HeuristicPolicy.__name__]['won']
    L2_diff = policy_eval_results[Level2HeuristicPolicy.__name__]['won'] - best_eval_results[Level2HeuristicPolicy.__name__]['won']

    # normalize duel_diff to account for the fact that it has fewer games
    total_duel_games = duel_result['won'] + duel_result['lost'] + duel_result['tied']
    total_eval_games = policy_eval_results[RandomPolicy.__name__]['won'] + \
                       policy_eval_results[RandomPolicy.__name__]['lost'] + \
                       policy_eval_results[RandomPolicy.__name__]['tied']
    scaling_factor = total_eval_games / (total_duel_games + 0.0)
    return (scaling_factor * duel_diff) + L0_diff + L1_diff + L2_diff


unique_states_visited = set()
all_transitions = []
for episode_num in range(1, num_episodes + 1):
    lr = gen_rate_step(episode_num, learning_rate_schedule)
    eps = gen_rate_exponential(episode_num, policy_epsilon_start, policy_epsilon_end, num_episodes, decay_speed)
    policy.set_boltzmann_action(False)
    policy.set_learning_rate(lr)
    policy.set_epsilon(eps)
    best.set_boltzmann_action(False)
    best.set_epsilon(best_epsilon)

    policy_actions = []
    opponent_actions = []
    policy_states = []
    opponent_states = []

    players = ['policy', 'best'] if episode_num % 2 == 0 else ['best', 'policy']
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
            edge = best.select_edge(board_state)
            opponent_actions.append(to_one_hot_action(board_state, edge))
            current_player, _ = game.select_edge(edge, current_player)
            unique_states_visited.add(as_string(game.get_board_state()))
    policy_reward = reward_fn.compute_reward(game, 'policy', 'best')
    opponent_reward = reward_fn.compute_reward(game, 'best', 'policy')

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

    # analyze results and save current policy weights to file
    if episode_num % num_episodes_per_policy_evaluation == 0:
        # play against opponents
        policy.set_boltzmann_action(False)
        policy.set_epsilon(0.0)
        best.set_boltzmann_action(False)
        best.set_epsilon(0.0)
        opponents = [RandomPolicy(), Level1HeuristicPolicy(board_size), Level2HeuristicPolicy(board_size)]

        duel_result = duel(board_size, policy, best)
        policy_eval_results = evaluate(policy, board_size, 1000, opponents)
        best_eval_results = evaluate(best, board_size, 1000, opponents)

        policy_improvement = compute_policy_improvement(duel_result, policy_eval_results, best_eval_results)

        if policy_improvement > 0:
            print("beat previous best (%s), duel: %s" % (policy_improvement, duel_result))
            best = policy.copy()

        results = policy_eval_results if policy_improvement > 0 else best_eval_results

        print("%s, %s, %s, %s, %s, %s, %s, %s" % (episode_num, results[RandomPolicy.__name__]['won'],
                                                  results[Level1HeuristicPolicy.__name__]['won'],
                                                  results[Level2HeuristicPolicy.__name__]['won'],
                                                  results, len(unique_states_visited), eps, lr))
        WeightWriter.print_episode(base_path, episode_num, policy.print_params)

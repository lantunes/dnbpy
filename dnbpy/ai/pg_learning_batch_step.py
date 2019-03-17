from dnbpy.ai import *
from dnbpy.util.helper_functions import *
from dnbpy.util.file_helper import *
from dnbpy.util.reward_util import *
from dnbpy.util.rate_util import *
from dnbpy.util.evaluator import *

board_size = (2, 2)
num_episodes = 2500000
batch_size = 32
temperature_schedule =   {0: 0.7,  500000: 0.6, 1000000: 0.5,  1500000: 0.4,  2000000: 0.3}
learning_rate_schedule = {0: 5e-3, 500000: 1e-3, 1000000: 5e-4, 1500000: 5e-5, 2000000: 1e-6}
use_symmetries = True
base_path = get_base_path_arg()

print("initializing for (%s, %s) game..." % (board_size[0], board_size[1]))

policy = PGPolicyCNN2(board_size, batch_size=batch_size)
reward_fn = DelayedBinaryReward()

print_info(board_size=board_size, num_episodes=num_episodes, policy=policy, mode='self-play', reward=reward_fn,
           updates='offline', temperature_schedule=temperature_schedule, learning_rate_schedule=learning_rate_schedule,
           architecture=policy.get_architecture(), batch_size=batch_size)

unique_states_visited = set()
all_transitions = []
for episode_num in range(1, num_episodes + 1):
    tmp = gen_rate_step(episode_num, temperature_schedule)
    lr = gen_rate_step(episode_num, learning_rate_schedule)
    policy.set_boltzmann_action(True)
    policy.set_temperature(tmp)
    policy.set_learning_rate(lr)
    players = [0, 1]
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
            edge = policy.select_edge(board_state)
            p0_actions.append(to_one_hot_action(board_state, edge))
            current_player, _ = game.select_edge(edge, 0)
            unique_states_visited.add(as_string(game.get_board_state()))
        else:
            p1_states.append(board_state)
            edge = policy.select_edge(board_state)
            p1_actions.append(to_one_hot_action(board_state, edge))
            current_player, _ = game.select_edge(edge, 1)
            unique_states_visited.add(as_string(game.get_board_state()))

    p0_reward = reward_fn.compute_reward(game, 0, 1)
    p1_reward = reward_fn.compute_reward(game, 1, 0)

    # don't add transitions that have 0 reward as the gradient will be zero anyways
    if p0_reward == 1:
        p0_outcomes = len(p0_actions)*[p0_reward]
        append_transitions(p0_states, p0_actions, p0_outcomes, all_transitions, use_symmetries, board_size)
    elif p1_reward == 1:
        p1_outcomes = len(p1_actions)*[p1_reward]
        append_transitions(p1_states, p1_actions, p1_outcomes, all_transitions, use_symmetries, board_size)

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
                                                  results, len(unique_states_visited), tmp, lr))
        WeightWriter.print_episode(base_path, episode_num, policy.print_params)
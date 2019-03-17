from dnbpy.ai import *
from dnbpy.util.helper_functions import *
from dnbpy.util.file_helper import *
from dnbpy.util.reward_util import *
from dnbpy.util.rate_util import *
from dnbpy.util.evaluator import *

board_size = (2, 2)
num_episodes = 1000000
learning_rate = 0.005
min_learning_rate = 0.000001
temperature = 0.2
batch_size = 32
decay_speed = 1.0
playout_schedule = {0: 100, 100000: 200, 200000: 300, 300000: 400, 400000: 500, 500000: 600, 700000: 700}
base_path = get_base_path_arg()
use_symmetries = True
mcts_prob = 0.5

print("initializing for (%s, %s) game..." % (board_size[0], board_size[1]))

policy = PGPolicyCNN2(board_size, batch_size=batch_size)
mcts_player = MCTSPolicy2(board_size, num_playouts=100)
reward_fn = DelayedBinaryReward()

print_info(board_size=board_size, num_episodes=num_episodes, policy=policy, mode='vs self and MCTS (schedule)',
           reward=reward_fn, updates='offline', learning_rate=learning_rate, min_learning_rate=min_learning_rate,
           temperature=temperature, architecture=policy.get_architecture(), batch_size=batch_size)

unique_states_visited = set()
all_transitions = []
games_played_against_mcts = 0

for episode_num in range(1, num_episodes + 1):
    lr = gen_rate_exponential(episode_num, learning_rate, min_learning_rate, num_episodes, decay_speed)
    tmp = temperature
    num_playouts = gen_rate_step(episode_num, playout_schedule)
    policy.set_boltzmann_action(True)
    policy.set_temperature(tmp)
    policy.set_learning_rate(lr)
    mcts_player.set_num_playouts(num_playouts)

    policy_actions = []
    opponent_actions = []
    policy_states = []
    opponent_states = []

    if random.random() < mcts_prob:
        players = ['policy', 'opponent'] if games_played_against_mcts % 2 == 0 else ['opponent', 'policy']
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
                edge = mcts_player.select_edge(board_state, game.get_score(current_player))
                opponent_actions.append(to_one_hot_action(board_state, edge))
                current_player, _ = game.select_edge(edge, current_player)
                unique_states_visited.add(as_string(game.get_board_state()))
        games_played_against_mcts += 1
        policy_reward = reward_fn.compute_reward(game, 'policy', 'opponent')
        opponent_reward = reward_fn.compute_reward(game, 'opponent', 'policy')
    else:
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

    # don't add transitions that have 0 reward as the gradient will be zero anyways
    if policy_reward == 1:
        policy_outcomes = len(policy_actions)*[policy_reward]
        append_transitions(policy_states, policy_actions, policy_outcomes, all_transitions, use_symmetries, board_size)
    elif opponent_reward == 1:
        opponent_outcomes = len(opponent_actions)*[opponent_reward]
        append_transitions(opponent_states, opponent_actions, opponent_outcomes, all_transitions, use_symmetries, board_size)

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

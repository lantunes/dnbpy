from ai import *
from dnbpy import *
from util.helper_functions import *
from util.file_helper import *
from util.reward_util import *
from util.rate_util import *
from util.evaluator import *
from util.duel import *
import concurrent.futures

board_size = (3, 3)
num_episodes = 1000000
learning_rate = 0.005
min_learning_rate = 0.005
batch_size = 32
decay_speed = 1.0
use_symmetries = True
num_episodes_per_policy_update = 100
num_episodes_per_search_policy_update = 100
# episodes_per_thread = [100]  # 1 core
# episodes_per_thread = [50, 50]  # 2 cores
# episodes_per_thread = [33, 33, 34]  # 4 cores
# episodes_per_thread = [14, 14, 14, 14, 14, 15, 15]  # 8 cores
# episodes_per_thread = [6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]  # 16 cores
# episodes_per_thread = [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2]  # 36 cores
# episodes_per_thread = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]  # 72 cores
episodes_per_thread = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] # 100 cores (even)
mcts_simulations = 1000
normalize_policy_probs_with_softmax = False
mcts_c = 5
activation = tf.nn.tanh
dropout_keep_prob = 1.0
base_path = get_base_path_arg()

print("initializing for (%s, %s) game..." % (board_size[0], board_size[1]))

curr_policy = PGPolicy3x3CNN(board_size, batch_size=batch_size, dropout_keep_prob=dropout_keep_prob, activation=activation)
curr_policy.set_boltzmann_action(False)
curr_policy.set_epsilon(0.0)
search_policy = PGPolicy3x3CNN(board_size, batch_size=batch_size, dropout_keep_prob=dropout_keep_prob, activation=activation)
search_policy.set_boltzmann_action(False)
search_policy.set_epsilon(0.0)
mcts_player = MCTSPolicyNetPolicyCpuct(board_size, num_playouts=mcts_simulations, cpuct=mcts_c,
                                       normalize_policy_probs_with_softmax=normalize_policy_probs_with_softmax)
reward_fn = DelayedBinaryReward()

print_info(board_size=board_size, num_episodes=num_episodes, policy=curr_policy, mode='MCTS ExIt', reward=reward_fn,
           updates='offline', learning_rate=learning_rate, min_learning_rate=min_learning_rate,
           architecture=curr_policy.get_architecture(), batch_size=batch_size, decay_speed=decay_speed,
           num_episodes_per_policy_update=num_episodes_per_policy_update, episodes_per_thread=episodes_per_thread,
           mcts=mcts_player, mcts_simulations=mcts_simulations, activation=activation, mcts_c=mcts_c)

unique_states_visited = set()
winning_transitions = []
search_transitions = []
current_episode_num = 0


def get_selected_index(child_state, parent_state):
    diff = [x1 - x2 for (x1, x2) in zip(child_state, parent_state)]
    argmax = max(enumerate(diff), key=lambda x: x[1])[0]
    return argmax


def append_prob_transitions(states, action_prob_maps, outcomes, all_transitions, use_symmetries):
    for i, _ in enumerate(states):
        state = states[i]

        action_prob_map = action_prob_maps[i]
        rows = board_size[0]
        cols = board_size[1]
        action_prob = [0.0]*((2*rows*cols) + rows + cols)
        for st in action_prob_map:
            indx = get_selected_index([int(i) for i in st], state)
            action_prob[indx] = action_prob_map[st]

        reward = outcomes[i]
        if use_symmetries:
            state_action_symmetries = to_state_action_pair_symmetries(board_size, state, action_prob)
            for symmetry in state_action_symmetries:
                all_transitions.append([symmetry[0], symmetry[1], reward])
        else:
            all_transitions.append([state, action_prob, reward])


def sample_edge(board_state, action_prob_map):
    choices = []
    probs = []
    for state in action_prob_map:
        choices.append(state)
        probs.append(action_prob_map[state])
    selected_state = np.random.choice(choices, p=probs)
    return get_selected_index([int(i) for i in selected_state], board_state)


def run_episodes(num_episodes_per_thread, search_params):
    search_pol = PGPolicy3x3CNN(board_size, batch_size=batch_size, existing_params=search_params, activation=activation)
    search_pol.set_boltzmann_action(False)
    search_pol.set_epsilon(0.0)
    mcts = MCTSPolicyNetPolicyCpuct(board_size, num_playouts=mcts_simulations, cpuct=mcts_c,
                                    normalize_policy_probs_with_softmax=normalize_policy_probs_with_softmax)

    action_transitions = []
    prob_transitions = []
    for episode_num in range(1, num_episodes_per_thread + 1):
        players = [0, 1] if episode_num % 2 == 0 else [1, 0]
        game = Game(board_size, players)
        current_player = game.get_current_player()

        p0_actions = []
        p1_actions = []
        p0_states = []
        p1_states = []
        p0_action_prob_maps = []
        p1_action_prob_maps = []

        while not game.is_finished():
            board_state = game.get_board_state()
            if current_player == 0:
                p0_states.append(board_state)
                action_prob_map = mcts.get_action_probs(board_state, game.get_score(current_player), search_pol)
                edge = sample_edge(board_state, action_prob_map)
                p0_action_prob_maps.append(action_prob_map)
                p0_actions.append(to_one_hot_action(board_state, edge))
                current_player, _ = game.select_edge(edge, current_player)
                unique_states_visited.add(as_string(game.get_board_state()))
            else:
                p1_states.append(board_state)
                action_prob_map = mcts.get_action_probs(board_state, game.get_score(current_player), search_pol)
                edge = sample_edge(board_state, action_prob_map)
                p1_action_prob_maps.append(action_prob_map)
                p1_actions.append(to_one_hot_action(board_state, edge))
                current_player, _ = game.select_edge(edge, current_player)
                unique_states_visited.add(as_string(game.get_board_state()))

        p0_reward = reward_fn.compute_reward(game, 0, 1)
        p1_reward = reward_fn.compute_reward(game, 1, 0)

        # don't add transitions that have 0 reward as the gradient will be zero anyways
        if p0_reward == 1:
            p0_outcomes = len(p0_actions)*[p0_reward]
            append_transitions(p0_states, p0_actions, p0_outcomes, action_transitions, use_symmetries, board_size)
        elif p1_reward == 1:
            p1_outcomes = len(p1_actions)*[p1_reward]
            append_transitions(p1_states, p1_actions, p1_outcomes, action_transitions, use_symmetries, board_size)
        p0_outcomes = len(p0_action_prob_maps)*[1]
        append_prob_transitions(p0_states, p0_action_prob_maps, p0_outcomes, prob_transitions, use_symmetries)
        p1_outcomes = len(p1_action_prob_maps)*[1]
        append_prob_transitions(p1_states, p1_action_prob_maps, p1_outcomes, prob_transitions, use_symmetries)
    return action_transitions, prob_transitions


while current_episode_num < num_episodes:
    search_params = search_policy.get_params()

    with concurrent.futures.ProcessPoolExecutor(max_workers=len(episodes_per_thread)) as executor:
        for r in executor.map(run_episodes, episodes_per_thread, [search_params]*len(episodes_per_thread)):
            winning_transitions.extend(r[0])
            search_transitions.extend(r[1])
    current_episode_num += sum(episodes_per_thread)

    lr = gen_rate_exponential(current_episode_num, learning_rate, min_learning_rate, num_episodes, decay_speed)
    curr_policy.set_learning_rate(lr)
    search_policy.set_learning_rate(lr)

    if current_episode_num % num_episodes_per_policy_update == 0:
        curr_policy.update_model(winning_transitions)
        winning_transitions = []

    if current_episode_num % num_episodes_per_search_policy_update == 0:
        search_policy.update_model(search_transitions)
        search_transitions = []

    # analyze results
    if current_episode_num % 1000 == 0:
        # play against opponents
        curr_policy.set_boltzmann_action(False)
        curr_policy.set_epsilon(0.0)
        opponents = [RandomPolicy(), Level1HeuristicPolicy(board_size), Level2HeuristicPolicy(board_size)]
        results = evaluate(curr_policy, board_size, 1000, opponents)
        print("%s, %s, %s, %s, %s, %s, %s, %s" % (current_episode_num, results[RandomPolicy.__name__]['won'],
                                                  results[Level1HeuristicPolicy.__name__]['won'],
                                                  results[Level2HeuristicPolicy.__name__]['won'],
                                                  results, len(unique_states_visited), 0.0, lr))
        WeightWriter.print_episode(base_path, current_episode_num, curr_policy.print_params)
        WeightWriter.print_episode(base_path, current_episode_num, search_policy.print_params, filename_prefix='search-weights-')

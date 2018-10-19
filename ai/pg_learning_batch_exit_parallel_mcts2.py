from ai import *
from dnbpy import *
from util.helper_functions import *
from util.file_helper import *
from util.rate_util import *
from util.evaluator import *
from util.duel import *
import concurrent.futures
import multiprocessing as mp

board_size = (3, 3)
num_iterations = 1000
learning_rate = 0.001
min_learning_rate = 0.00005
batch_size = 32
decay_speed = 1.0
use_symmetries = True
episodes_per_worker = 2
num_workers = 100
episodes_per_iteration = episodes_per_worker * num_workers
mcts_simulations = 1000
normalize_policy_probs_with_softmax = False
mcts_c = 5
activation = tf.nn.relu
dropout_keep_prob = 1.0
base_path = get_base_path_arg()


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


def run_episodes(num_episodes, search_params):
    search_pol = PGPolicy3x3CNN(board_size, batch_size=batch_size, existing_params=search_params, activation=activation)
    search_pol.set_boltzmann_action(False)
    search_pol.set_epsilon(0.0)
    mcts = MCTSPolicyNetPolicyCpuct(board_size, num_playouts=mcts_simulations, cpuct=mcts_c,
                                    normalize_policy_probs_with_softmax=normalize_policy_probs_with_softmax)

    prob_transitions = []
    for episode_num in range(1, num_episodes + 1):
        players = [0, 1] if episode_num % 2 == 0 else [1, 0]
        game = Game(board_size, players)
        current_player = game.get_current_player()

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
                current_player, _ = game.select_edge(edge, current_player)
            else:
                p1_states.append(board_state)
                action_prob_map = mcts.get_action_probs(board_state, game.get_score(current_player), search_pol)
                edge = sample_edge(board_state, action_prob_map)
                p1_action_prob_maps.append(action_prob_map)
                current_player, _ = game.select_edge(edge, current_player)

        p0_outcomes = len(p0_action_prob_maps)*[1]
        append_prob_transitions(p0_states, p0_action_prob_maps, p0_outcomes, prob_transitions, use_symmetries)
        p1_outcomes = len(p1_action_prob_maps)*[1]
        append_prob_transitions(p1_states, p1_action_prob_maps, p1_outcomes, prob_transitions, use_symmetries)
    return prob_transitions

if __name__ == '__main__':
    mp.set_start_method('spawn')

    print("initializing for (%s, %s) game..." % (board_size[0], board_size[1]))

    search_policy = PGPolicy3x3CNN(board_size, batch_size=batch_size, dropout_keep_prob=dropout_keep_prob, activation=activation)
    search_policy.set_boltzmann_action(False)
    search_policy.set_epsilon(0.0)
    mcts_player = MCTSPolicyNetPolicyCpuct(board_size, num_playouts=mcts_simulations, cpuct=mcts_c,
                                           normalize_policy_probs_with_softmax=normalize_policy_probs_with_softmax)

    print_info(board_size=board_size, num_iterations=num_iterations, policy=search_policy, mode='MCTS ExIt',
           updates='offline', learning_rate=learning_rate, min_learning_rate=min_learning_rate,
           architecture=search_policy.get_architecture(), batch_size=batch_size, decay_speed=decay_speed,
           episodes_per_iteration=episodes_per_iteration, mcts=mcts_player, mcts_simulations=mcts_simulations,
           activation=activation, mcts_c=mcts_c, episodes_per_worker=episodes_per_worker, num_workers=num_workers)

    current_iteration_num = 0
    search_transitions = []

    while current_iteration_num < num_iterations:
        search_params = search_policy.get_params()

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            for r in executor.map(run_episodes, [episodes_per_worker]*num_workers, [search_params]*num_workers):
                search_transitions.extend(r)
        current_iteration_num += 1

        lr = gen_rate_exponential(current_iteration_num, learning_rate, min_learning_rate, num_iterations, decay_speed)
        search_policy.set_learning_rate(lr)

        # update the model
        search_policy.update_model(search_transitions)
        search_transitions = []

        # analyze model; play against opponents
        search_policy.set_boltzmann_action(False)
        search_policy.set_epsilon(0.0)
        opponents = [RandomPolicy(), Level1HeuristicPolicy(board_size), Level2HeuristicPolicy(board_size)]
        results = evaluate_parallel(search_policy, board_size, 1000, opponents)
        print("%s, %s, %s, %s, %s, %s" % (current_iteration_num, results[RandomPolicy.__name__]['won'],
                                          results[Level1HeuristicPolicy.__name__]['won'],
                                          results[Level2HeuristicPolicy.__name__]['won'],
                                          results, lr))
        WeightWriter.print_episode(base_path, current_iteration_num, search_policy.print_params)

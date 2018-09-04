from ai import *
from dnbpy import *
from util.helper_functions import *
from util.file_helper import *
from util.reward_util import *
from util.rate_util import *
from util.evaluator import *
import concurrent.futures

board_size = (2, 2)
num_episodes = 1000000
learning_rate = 0.005
min_learning_rate = 0.005
temperature_schedule = {0: 0.67}
batch_size = 32
decay_speed = 1.0
use_symmetries = True
num_episodes_per_policy_update = 100
# episodes_per_thread = [100]  # 1 core
# episodes_per_thread = [50, 50]  # 2 cores
# episodes_per_thread = [33, 33, 34]  # 4 cores
# episodes_per_thread = [14, 14, 14, 14, 14, 15, 15]  # 8 cores
# episodes_per_thread = [6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]  # 16 cores
episodes_per_thread = [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2]  # 36 cores
mcts_simulations = 100
mcts_c = 5
normalize_policy_probs_with_softmax = True
activation = tf.nn.tanh
base_path = get_base_path_arg()

print("initializing for (%s, %s) game..." % (board_size[0], board_size[1]))

curr_policy = PGPolicyCNN2(board_size, batch_size=batch_size, activation=activation)
curr_policy.set_boltzmann_action(False)
curr_policy.set_epsilon(0.0)
mcts_player = MCTSPolicyNetPolicyCpuct(board_size, num_playouts=mcts_simulations, cpuct=mcts_c,
                                       normalize_policy_probs_with_softmax=normalize_policy_probs_with_softmax)
reward_fn = DelayedBinaryReward()

print_info(board_size=board_size, num_episodes=num_episodes, policy=curr_policy, mode='MCTS ExIt', reward=reward_fn,
           updates='offline', learning_rate=learning_rate, min_learning_rate=min_learning_rate,
           architecture=curr_policy.get_architecture(), batch_size=batch_size, decay_speed=decay_speed,
           num_episodes_per_policy_update=num_episodes_per_policy_update, episodes_per_thread=episodes_per_thread,
           mcts=mcts_player, mcts_simulations=mcts_simulations, mcts_c=mcts_c, activation=activation,
           temperature_schedule=temperature_schedule)

unique_states_visited = set()
all_transitions = []
current_episode_num = 0


def sample_edge(board_state, action_prob_map):
    def get_selected_index(child_state, parent_state):
        diff = [x1 - x2 for (x1, x2) in zip(child_state, parent_state)]
        argmax = max(enumerate(diff), key=lambda x: x[1])[0]
        return argmax
    choices = []
    probs = []
    for state in action_prob_map:
        choices.append(state)
        probs.append(action_prob_map[state])
    selected_state = np.random.choice(choices, p=probs)
    return get_selected_index([int(i) for i in selected_state], board_state)


def run_episodes(num_episodes_per_thread, existing_params, temperature):
    pol = PGPolicyCNN2(board_size, batch_size=batch_size, existing_params=existing_params, activation=activation)
    run_transitions = []
    for episode_num in range(1, num_episodes_per_thread + 1):
        players = [0, 1] if episode_num % 2 == 0 else [1, 0]
        game = Game(board_size, players)
        current_player = game.get_current_player()
        mcts = MCTSPolicyNetPolicyCpuct(board_size, num_playouts=mcts_simulations, cpuct=mcts_c,
                                        normalize_policy_probs_with_softmax=normalize_policy_probs_with_softmax)

        p0_actions = []
        p1_actions = []
        p0_states = []
        p1_states = []

        while not game.is_finished():
            board_state = game.get_board_state()
            if current_player == 0:
                p0_states.append(board_state)
                action_prob_map = mcts.get_action_probs(board_state, game.get_score(current_player), pol, temperature)
                edge = sample_edge(board_state, action_prob_map)
                p0_actions.append(to_one_hot_action(board_state, edge))
                current_player, _ = game.select_edge(edge, 0)
                unique_states_visited.add(as_string(game.get_board_state()))
            else:
                p1_states.append(board_state)
                action_prob_map = mcts.get_action_probs(board_state, game.get_score(current_player), pol, temperature)
                edge = sample_edge(board_state, action_prob_map)
                p1_actions.append(to_one_hot_action(board_state, edge))
                current_player, _ = game.select_edge(edge, 1)
                unique_states_visited.add(as_string(game.get_board_state()))

        p0_reward = reward_fn.compute_reward(game, 0, 1)
        p1_reward = reward_fn.compute_reward(game, 1, 0)

        # don't add transitions that have 0 reward as the gradient will be zero anyways
        if p0_reward == 1:
            p0_outcomes = len(p0_actions)*[p0_reward]
            append_transitions(p0_states, p0_actions, p0_outcomes, run_transitions, use_symmetries, board_size)
        elif p1_reward == 1:
            p1_outcomes = len(p1_actions)*[p1_reward]
            append_transitions(p1_states, p1_actions, p1_outcomes, run_transitions, use_symmetries, board_size)
    return run_transitions


while current_episode_num < num_episodes:
    existing_params = curr_policy.get_params()
    tmp = gen_rate_step(current_episode_num, temperature_schedule)
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(episodes_per_thread)) as executor:
        for r in executor.map(run_episodes, episodes_per_thread, [existing_params]*len(episodes_per_thread), [tmp]*len(episodes_per_thread)):
            all_transitions.extend(r)
    current_episode_num += sum(episodes_per_thread)

    lr = gen_rate_exponential(current_episode_num, learning_rate, min_learning_rate, num_episodes, decay_speed)
    curr_policy.set_learning_rate(lr)

    if current_episode_num % num_episodes_per_policy_update == 0:
        curr_policy.update_model(all_transitions)
        all_transitions = []

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
                                                  results, len(unique_states_visited), tmp, lr))
        WeightWriter.print_episode(base_path, current_episode_num, curr_policy.print_params)

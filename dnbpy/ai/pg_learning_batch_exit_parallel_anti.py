from dnbpy.ai import *
from dnbpy.util.helper_functions import *
from dnbpy.util.file_helper import *
from dnbpy.util.reward_util import *
from dnbpy.util.rate_util import *
from dnbpy.util.evaluator import *
import concurrent.futures

board_size = (2, 2)
num_episodes = 1000000
learning_rate = 0.005
min_learning_rate = 0.005
temperature = 1.0
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
normalize_combined_with_softmax = False
base_path = get_base_path_arg()

print("initializing for (%s, %s) game..." % (board_size[0], board_size[1]))

policy = PGPolicyCNN2(board_size, batch_size=batch_size, activation=activation)
anti_policy = PGPolicyCNN2(board_size, batch_size=batch_size, activation=activation)
policy.set_boltzmann_action(False)
policy.set_epsilon(0.0)
anti_policy.set_boltzmann_action(False)
anti_policy.set_epsilon(0.0)
mcts_player = MCTSPolicyNetPolicyCpuct(board_size, num_playouts=mcts_simulations, cpuct=mcts_c,
                                       normalize_policy_probs_with_softmax=normalize_policy_probs_with_softmax)
reward_fn = DelayedBinaryReward()

print_info(board_size=board_size, num_episodes=num_episodes, policy=policy, mode='MCTS ExIt', reward=reward_fn,
           updates='offline', learning_rate=learning_rate, min_learning_rate=min_learning_rate, temperature=temperature,
           architecture=policy.get_architecture(), batch_size=batch_size, decay_speed=decay_speed,
           num_episodes_per_policy_update=num_episodes_per_policy_update, episodes_per_thread=episodes_per_thread,
           mcts=mcts_player, mcts_simulations=mcts_simulations, mcts_c=mcts_c,
           normalize_action_probs_with_softmax=normalize_policy_probs_with_softmax, activation=activation)


class CombinedPolicy:
    def __init__(self, pol, anti_pol, normalize_combined_with_softmax):
        self._pol = pol
        self._anti_pol = anti_pol
        self._normalize_combined_with_softmax = normalize_combined_with_softmax

    def get_action_probs(self, st, normalize_with_softmax):
        policy_prob_map = self._pol.get_action_probs(st, normalize_with_softmax)
        anti_policy_prob_map = self._anti_pol.get_action_probs(st, normalize_with_softmax)
        diff_prob_map = {}
        for state in policy_prob_map:
            diff_prob_map[state] = policy_prob_map[state] - anti_policy_prob_map[state]
        if len(diff_prob_map) == 1:
            for s in diff_prob_map:
                diff_prob_map[s] = 1.0
            return diff_prob_map
        if self._normalize_combined_with_softmax:
            exp_sum = np.sum(np.exp(np.array(list(diff_prob_map.values())) / temperature))
            for state in diff_prob_map:
                diff_prob_map[state] = np.exp(diff_prob_map[state] / temperature) / exp_sum
        return diff_prob_map


unique_states_visited = set()
all_transitions = []
all_anti_transitions = []
current_episode_num = 0


def run_episodes(num_episodes_per_thread, existing_policy_params, existing_anti_policy_params):
    local_policy = PGPolicyCNN2(board_size, batch_size=batch_size, existing_params=existing_policy_params, activation=activation)
    local_anti_policy = PGPolicyCNN2(board_size, batch_size=batch_size, existing_params=existing_anti_policy_params, activation=activation)
    pol = CombinedPolicy(local_policy, local_anti_policy, normalize_combined_with_softmax)
    run_transitions = []
    run_anti_transitions = []
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
                edge = mcts.select_edge(board_state, game.get_score(current_player), pol)
                p0_actions.append(to_one_hot_action(board_state, edge))
                current_player, _ = game.select_edge(edge, 0)
                unique_states_visited.add(as_string(game.get_board_state()))
            else:
                p1_states.append(board_state)
                edge = mcts.select_edge(board_state, game.get_score(current_player), pol)
                p1_actions.append(to_one_hot_action(board_state, edge))
                current_player, _ = game.select_edge(edge, 1)
                unique_states_visited.add(as_string(game.get_board_state()))

        p0_reward = reward_fn.compute_reward(game, 0, 1)
        p1_reward = reward_fn.compute_reward(game, 1, 0)

        if p0_reward == 1:
            p0_outcomes = len(p0_actions)*[p0_reward]
            append_transitions(p0_states, p0_actions, p0_outcomes, run_transitions, use_symmetries, board_size)
            p1_outcomes = len(p1_actions)*[p0_reward]
            append_transitions(p1_states, p1_actions, p1_outcomes, run_anti_transitions, use_symmetries, board_size)
        elif p1_reward == 1:
            p1_outcomes = len(p1_actions)*[p1_reward]
            append_transitions(p1_states, p1_actions, p1_outcomes, run_transitions, use_symmetries, board_size)
            p0_outcomes = len(p0_actions)*[p1_reward]
            append_transitions(p0_states, p0_actions, p0_outcomes, run_anti_transitions, use_symmetries, board_size)
    return run_transitions, run_anti_transitions


while current_episode_num < num_episodes:
    existing_policy_params = policy.get_params()
    existing_anti_policy_params = anti_policy.get_params()
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(episodes_per_thread)) as executor:
        for r, a_r in executor.map(run_episodes, episodes_per_thread, [existing_policy_params]*len(episodes_per_thread),
                                   [existing_anti_policy_params]*len(episodes_per_thread)):
            all_transitions.extend(r)
            all_anti_transitions.extend(a_r)
    current_episode_num += sum(episodes_per_thread)

    lr = gen_rate_exponential(current_episode_num, learning_rate, min_learning_rate, num_episodes, decay_speed)
    policy.set_learning_rate(lr)
    anti_policy.set_learning_rate(lr)

    if current_episode_num % num_episodes_per_policy_update == 0:
        policy.update_model(all_transitions)
        anti_policy.update_model(all_anti_transitions)
        all_transitions = []
        all_anti_transitions = []

    # analyze results
    if current_episode_num % 1000 == 0:
        # play against opponents
        policy.set_boltzmann_action(False)
        policy.set_epsilon(0.0)
        opponents = [RandomPolicy(), Level1HeuristicPolicy(board_size), Level2HeuristicPolicy(board_size)]
        results = evaluate(policy, board_size, 1000, opponents)
        print("%s, %s, %s, %s, %s, %s, %s, %s" % (current_episode_num, results[RandomPolicy.__name__]['won'],
                                                  results[Level1HeuristicPolicy.__name__]['won'],
                                                  results[Level2HeuristicPolicy.__name__]['won'],
                                                  results, len(unique_states_visited), 1.0, lr))
        WeightWriter.print_episode(base_path, current_episode_num, policy.print_params)

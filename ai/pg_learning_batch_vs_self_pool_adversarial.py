from ai import *
from dnbpy import *
from util.helper_functions import *
from util.file_helper import *
from util.reward_util import *
from util.rate_util import *
from util.evaluator import *
from util.opponent_pool_util import *

board_size = (2, 2)
num_episodes = 1000000
learning_rate_schedule = {0: 0.005}
# temperature_schedule = {0: 0.9, 10000: 0.85, 20000: 0.8, 30000: 0.75, 40000: 0.65, 50000: 0.55, 60000: 0.45, 70000: 0.35}
temperature_schedule = {0: 2.0, 50000: 1.5, 100000: 1.0, 150000: 0.9, 200000: 0.8, 250000: 0.75, 300000: 0.7, 350000: 0.65}
batch_size = 32
decay_speed = 1.0
opponent_pool_max_size = 100
num_episodes_per_update = 500
dropout_keep_prob = 1.0
use_symmetries = False
normalize_action_probs_with_softmax = False
base_path = get_base_path_arg()

print("initializing for (%s, %s) game..." % (board_size[0], board_size[1]))

policy = PGPolicyCNN2Adversarial(board_size, batch_size=batch_size, dropout_keep_prob=dropout_keep_prob)
# anti_policy = PGPolicyCNN2(board_size, batch_size=batch_size, dropout_keep_prob=dropout_keep_prob)
anti_policy = PGPolicyCNN2Adversarial(board_size, batch_size=batch_size, dropout_keep_prob=dropout_keep_prob)
anti_policy.set_boltzmann_action(False)
anti_policy.set_epsilon(0.0)
opponent = policy
reward_fn = DelayedBinaryReward()
opponent_pool = OpponentPool(max_size=opponent_pool_max_size)

print_info(board_size=board_size, num_episodes=num_episodes, policy=policy, mode='self-play pool', reward=reward_fn,
           updates='offline', learning_rate_schedule=learning_rate_schedule, temperature_schedule=temperature_schedule,
           architecture=policy.get_architecture(), batch_size=batch_size, decay_speed=decay_speed,
           dropout_keep_prob=dropout_keep_prob, use_symmetries=use_symmetries,
           num_episodes_per_policy_update=num_episodes_per_update,
           num_episodes_per_opponent_cache=num_episodes_per_update, opponent_pool_max_size=opponent_pool_max_size,
           normalize_action_probs_with_softmax=normalize_action_probs_with_softmax)


def get_selected_index(child_state, parent_state):
    diff = [x1 - x2 for (x1, x2) in zip(child_state, parent_state)]
    argmax = max(enumerate(diff), key=lambda x: x[1])[0]
    return argmax


def to_board_state_probs(board_state, action_probs_map):
    action_probs = [0.0]*len(board_state)
    for child_state in action_probs_map:
        child_state_index = get_selected_index([int(i) for i in child_state], board_state)
        action_probs[child_state_index] = action_probs_map[child_state]
    return action_probs


def sample_action(action_probs):
    indices = list(range(len(action_probs)))
    return np.random.choice(indices, p=action_probs)


def append_adversarial_transitions(states, actions, all_transitions, adversarial_policy):
    for i, _ in enumerate(actions):
        state = states[i]
        action = actions[i]
        anti_probs_map = adversarial_policy.get_action_probs(state, normalize_with_softmax=normalize_action_probs_with_softmax)
        anti_probs = [0.0]*len(action)
        # get the probs
        for child_state in anti_probs_map:
            child_state_index = get_selected_index([int(i) for i in child_state], state)
            # NOTE: we have to do the "1 - anti_probs" here
            anti_probs[child_state_index] = 1.0 - anti_probs_map[child_state] if anti_probs_map[child_state] < 1.0 else 1.0
        # re-normalize
        sum_anti_probs = sum(anti_probs)
        for idx in range(len(anti_probs)):
            anti_probs[idx] = anti_probs[idx] / sum_anti_probs

        if use_symmetries:
            state_action_symmetries = to_state_action_pair_symmetries(board_size, state, action)
            state_anti_prob_symmetries = to_state_action_pair_symmetries(board_size, state, anti_probs)
            for i in range(len(state_action_symmetries)):
                all_transitions.append([state_action_symmetries[i][0], state_action_symmetries[i][1], state_anti_prob_symmetries[i][1]])
        else:
            all_transitions.append([state, action, anti_probs])


unique_states_visited = set()
winning_transitions = []
losing_transitions = []
for episode_num in range(1, num_episodes + 1):
    tmp = gen_rate_step(episode_num, temperature_schedule)
    lr = gen_rate_step(episode_num, learning_rate_schedule)
    policy.set_boltzmann_action(False)
    policy.set_learning_rate(lr)
    anti_policy.set_learning_rate(lr)

    policy_actions = []
    opponent_actions = []
    policy_states = []
    opponent_states = []

    players = ['policy', 'opponent'] if episode_num % 2 == 0 else ['opponent', 'policy']
    game = Game(board_size, players)
    current_player = game.get_current_player()
    while not game.is_finished():
        board_state = game.get_board_state()
        if current_player == 'policy':
            policy_states.append(board_state)
            # edge = policy.select_edge(board_state)
            # policy_actions.append(to_one_hot_action(board_state, edge))
            action_probs_map = policy.get_action_probs(board_state, normalize_with_softmax=normalize_action_probs_with_softmax, temperature=tmp)
            action_probs = to_board_state_probs(board_state, action_probs_map)
            policy_actions.append(action_probs)
            edge = sample_action(action_probs)
            current_player, _ = game.select_edge(edge, current_player)
            unique_states_visited.add(as_string(game.get_board_state()))
        else:
            opponent_states.append(board_state)
            # edge = opponent.select_edge(board_state)
            # opponent_actions.append(to_one_hot_action(board_state, edge))
            action_probs_map = opponent.get_action_probs(board_state, normalize_with_softmax=normalize_action_probs_with_softmax, temperature=tmp)
            action_probs = to_board_state_probs(board_state, action_probs_map)
            opponent_actions.append(action_probs)
            edge = sample_action(action_probs)
            current_player, _ = game.select_edge(edge, current_player)
            unique_states_visited.add(as_string(game.get_board_state()))
    policy_reward = reward_fn.compute_reward(game, 'policy', 'opponent')
    opponent_reward = reward_fn.compute_reward(game, 'opponent', 'policy')

    if policy_reward == 1:
        append_adversarial_transitions(policy_states, policy_actions, winning_transitions, anti_policy)
        # opponent_outcomes = len(opponent_actions)*[policy_reward]
        # append_transitions(opponent_states, opponent_actions, opponent_outcomes, losing_transitions, use_symmetries, board_size)
        append_adversarial_transitions(opponent_states, opponent_actions, losing_transitions, policy)
    elif opponent_reward == 1:
        append_adversarial_transitions(opponent_states, opponent_actions, winning_transitions, anti_policy)
        # policy_outcomes = len(policy_actions)*[opponent_reward]
        # append_transitions(policy_states, policy_actions, policy_outcomes, losing_transitions, use_symmetries, board_size)
        append_adversarial_transitions(policy_states, policy_actions, losing_transitions, policy)

    if episode_num % num_episodes_per_update == 0:
        policy.update_model(winning_transitions)
        anti_policy.update_model(losing_transitions)
        opponent_pool.add_to_pool(policy.copy())
        winning_transitions = []
        losing_transitions = []
        # TODO do we want to keep the original epsilon or update the opponent to use the latest epsilon?
        opponent = opponent_pool.sample_opponent()

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
                                                               results, len(unique_states_visited), tmp, lr))
        WeightWriter.print_episode(base_path, episode_num, policy.print_params)
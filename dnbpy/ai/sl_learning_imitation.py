from dnbpy.ai import *
from dnbpy.util.helper_functions import *
from dnbpy.util.file_helper import *
from dnbpy.util.reward_util import *
from dnbpy.util.rate_util import *
from dnbpy.util.evaluator import *

board_size = (3, 3)
num_episodes = 1000000
learning_rate_schedule = {0: 0.005}
epsilon = 0.0
batch_size = 32
decay_speed = 1.0
use_symmetries = True
dropout_keep_prob = 1.0
num_episodes_per_policy_update = 1000
activation = tf.nn.relu
base_path = get_base_path_arg()

print("initializing for (%s, %s) game..." % (board_size[0], board_size[1]))

policy = SLValue3x3CNN(board_size, batch_size=batch_size, dropout_keep_prob=dropout_keep_prob, activation=activation)
policy.set_epsilon(0.0)
L2 = Level2HeuristicPolicy(board_size)
L1 = Level1HeuristicPolicy(board_size)
L0 = RandomPolicy()
reward_fn = DelayedBinaryReward()

print_info(board_size=board_size, num_episodes=num_episodes, policy=policy, mode='watch opponents',
           reward=reward_fn, updates='offline', learning_rate_schedule=learning_rate_schedule,
           epsilon=epsilon, architecture=policy.get_architecture(), batch_size=batch_size,
           dropout_keep_prob=dropout_keep_prob, num_episodes_per_policy_update=num_episodes_per_policy_update)

unique_states_visited = set()
all_transitions = []

for episode_num in range(1, num_episodes + 1):
    lr = gen_rate_step(episode_num, learning_rate_schedule)
    policy.set_learning_rate(lr)

    p0_states = []
    p1_states = []
    p0 = random.choice([L0, L2]) #[L0, L1, L2])
    p1 = random.choice([L0, L2]) #[L0, L1, L2])
    players = [0, 1]
    game = Game(board_size, players)
    current_player = game.get_current_player()
    while not game.is_finished():
        board_state = game.get_board_state()
        if current_player == 0:
            p0_states.append(board_state)
            edge = p0.select_edge(board_state)
            current_player, _ = game.select_edge(edge, 0)
            unique_states_visited.add(as_string(game.get_board_state()))
        else:
            p1_states.append(board_state)
            edge = p1.select_edge(board_state)
            current_player, _ = game.select_edge(edge, 1)
            unique_states_visited.add(as_string(game.get_board_state()))

    p0_reward = reward_fn.compute_reward(game, 0, 1)
    p1_reward = reward_fn.compute_reward(game, 1, 0)
    p0_outcomes = len(p0_states)*[p0_reward]
    append_value_transitions(p0_states, p0_outcomes, all_transitions, use_symmetries, board_size)
    p1_outcomes = len(p1_states)*[p1_reward]
    append_value_transitions(p1_states, p1_outcomes, all_transitions, use_symmetries, board_size)

    if episode_num % num_episodes_per_policy_update == 0:
        policy.update_model(all_transitions)
        all_transitions = []

    # analyze results
    if episode_num % 1000 == 0:
        # play against opponents
        policy.set_epsilon(0.0)
        opponents = [RandomPolicy(), Level1HeuristicPolicy(board_size), Level2HeuristicPolicy(board_size)]
        results = evaluate(policy, board_size, 1000, opponents)
        print("%s, %s, %s, %s, %s, %s, %s, %s" % (episode_num, results[RandomPolicy.__name__]['won'],
                                                  results[Level1HeuristicPolicy.__name__]['won'],
                                                  results[Level2HeuristicPolicy.__name__]['won'],
                                                  results, len(unique_states_visited), epsilon, lr))
        WeightWriter.print_episode(base_path, episode_num, policy.print_params)

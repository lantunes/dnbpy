import sys
sys.path.append("/Users/u6042446/Desktop/dnbpy/")
from ai import *
from dnbpy import *
from util.helper_functions import *
from util.file_helper import *
from util.reward_util import *
from util.rate_util import *
from util.evaluator import *
from util.opponent_pool_util import *
from util.duel import *
from util.conf import *
from util.initializer_util import read_params

conf = conf()

num_input_channels = 1
results_file_base_path = conf._results_base_path
experiment_name = "/adver_shapedReward_epsilonGreedy_"+str(num_input_channels)+"InputChannels_"
#create required files
self_play_metric_file = open(results_file_base_path+experiment_name+"self_play_metric.txt","w")
benchmark_play_metric_file = open(results_file_base_path+experiment_name+"benchmark_metric.txt","w")
weights_file = open(results_file_base_path+experiment_name+"weights.txt","w")

board_size = (3, 3)
num_episodes = 1000000
learning_rate_schedule = {0: 0}
anti_policy_rate_schedule = {0: 0,100000:.1,200000:.01,300000:0}
l_max = .005
l_min = .000005
epsilon_schedule = {0: 0.2, 10000: 0.85, 20000: 0.8, 30000: 0.75, 40000: 0.65, 50000: 0.55, 60000: 0.45, 70000: 0.35}
batch_size = 32
decay_speed = 1.0
opponent_pool_max_size = 100
num_episodes_per_update = 500
dropout_keep_prob = 1.0
use_symmetries = True
normalize_action_probs_with_softmax = False
#base_path = get_base_path_arg()
base_path = "/Users/u6042446/Desktop/new_Luis/dnbpy/ai/"

print("initializing for (%s, %s) game..." % (board_size[0], board_size[1]))

pg_params = read_params("Results/adver_shapedReward_epsilonGreedy_1InputChannels__weights_300000.txt")

policy = PGPolicy3x3CNN(board_size, n_channels=num_input_channels,batch_size=batch_size, dropout_keep_prob=dropout_keep_prob,existing_params=pg_params)
anti_policy = PGPolicy3x3CNN(board_size, n_channels=num_input_channels,batch_size=batch_size, dropout_keep_prob=dropout_keep_prob)
policy.set_boltzmann_action(False)
anti_policy.set_boltzmann_action(False)
opponent = policy.copy()
anti_opponent = anti_policy.copy()
reward_fn = DelayedBinaryReward()
reward_fn_shaped = DelayedShapedReward()
opponent_pool = OpponentPool(max_size=opponent_pool_max_size)

print_info(board_size=board_size, num_episodes=num_episodes, policy=policy, mode='self-play pool', reward=reward_fn,
           updates='offline', learning_rate_schedule=learning_rate_schedule, epsilon_schedule=epsilon_schedule,
           architecture=policy.get_architecture(), batch_size=batch_size, decay_speed=decay_speed,
           dropout_keep_prob=dropout_keep_prob, use_symmetries=use_symmetries,
           num_episodes_per_policy_update=num_episodes_per_update,
           num_episodes_per_opponent_cache=num_episodes_per_update, opponent_pool_max_size=opponent_pool_max_size)


def select_edge_with_anti(pol, anti_pol, board_state,tensor, epsilon,anti_policy_rate):
    if random.random() < epsilon:
        zero_indices = []  # indices of legal actions
        for i in range(len(board_state)):
            if board_state[i] == 0:
                zero_indices.append(i)
        return random.choice(zero_indices)

    def get_selected_index(child_state, parent_state):
        diff = [x1 - x2 for (x1, x2) in zip(child_state, parent_state)]
        argmax = max(enumerate(diff), key=lambda x: x[1])[0]
        return argmax
    policy_prob_map = pol.get_action_probs_input_channel(board_state,tensor, normalize_action_probs_with_softmax)
    anti_policy_prob_map = anti_pol.get_action_probs_input_channel(board_state,tensor, normalize_action_probs_with_softmax)
    diff_prob_map = {}

    states = []
    probs = []
    for state in policy_prob_map:
        diff_prob_map[state] = policy_prob_map[state] - anti_policy_rate*anti_policy_prob_map[state]
        states.append(state)
        probs.append(diff_prob_map[state])

    probs = np.array(probs)/sum(probs)
    if anti_policy_rate>0:
        highest_prob = None
        selected_state = None
        for state, val in diff_prob_map.items():
            if highest_prob is None or val > highest_prob:
                highest_prob = val
                selected_state = state
    else:
        selected_index = np.random.multinomial(1, probs).argmax()
        selected_state = states[selected_index]

    return get_selected_index([int(i) for i in selected_state], board_state)


class PolicyForEvaluation(Policy):
    def __init__(self, pol, anti_pol,anti_policy_rate):
        self._pol = pol
        self._anti_pol = anti_pol
        self._anti_policy_rate = anti_policy_rate

    def select_edge(self, st,tensor):
        return select_edge_with_anti(self._pol, self._anti_pol, st,tensor, 0.0,self._anti_policy_rate)


unique_states_visited = set()
all_transitions = []
all_anti_transitions = []
all_returns = []

for episode_num in range(1, num_episodes + 1):
    epsilon = gen_rate_step(episode_num, epsilon_schedule)
    #lr = gen_rate_step(episode_num, learning_rate_schedule)
    anti_policy_rate = gen_rate_step(episode_num, anti_policy_rate_schedule)
    lr = gen_rate_exponential(episode_num,l_max,l_min,num_episodes) #variable-rate learning
    lr = 0

    policy.set_epsilon(epsilon)
    policy.set_learning_rate(lr)
    anti_policy.set_epsilon(epsilon)
    anti_policy.set_learning_rate(lr)

    policy_actions = []
    opponent_actions = []
    policy_states = []
    opponent_states = []

    players = ['policy', 'opponent'] if episode_num % 2 == 0 else ['opponent', 'policy']
    game = Game_With_Box(board_size, players)
    current_player = game.get_current_player()

    policy_imm_reward_list = []
    opponent_imm_reward_list = []

    while not game.is_finished():
        board_state = game.get_board_state()
        edge_matrix = convert_board_state_to_edge_matrix(board_size,board_state)



        if current_player == 'policy':
            #policy_states.append(board_state)
            #policy_states.append(edge_matrix)
            tensor = game.get_tensor_representation(current_player,num_input_channels)
            policy_states.append(tensor)
            edge = select_edge_with_anti(policy, anti_policy, board_state, tensor, epsilon,anti_policy_rate)
            policy_actions.append(to_one_hot_action(board_state, edge))
            current_player, boxes_made = game.select_edge(edge, current_player)
            if len(opponent_imm_reward_list) > 0: opponent_imm_reward_list[-1] -= boxes_made
            unique_states_visited.add(as_string(game.get_board_state()))
            policy_imm_reward_list.append(boxes_made)
        else:
            #opponent_states.append(board_state)
            #opponent_states.append(edge_matrix)
            tensor = game.get_tensor_representation(current_player,num_input_channels)
            opponent_states.append(tensor)
            edge = select_edge_with_anti(opponent, anti_opponent, board_state,tensor, epsilon,anti_policy_rate)
            opponent_actions.append(to_one_hot_action(board_state, edge))
            current_player, boxes_made = game.select_edge(edge, current_player)
            if len(policy_imm_reward_list) > 0: policy_imm_reward_list[-1] -= boxes_made
            unique_states_visited.add(as_string(game.get_board_state()))
            opponent_imm_reward_list.append(boxes_made)


    policy_reward = reward_fn.compute_reward(game, 'policy', 'opponent')
    opponent_reward = reward_fn.compute_reward(game, 'opponent', 'policy')
    shaped_policy_rewad = reward_fn_shaped.compute_reward(game, 'policy', 'opponent')
    shaped_opponent_reward = reward_fn_shaped.compute_reward(game, 'opponent', 'policy')


    if policy_reward == 1:
        policy_outcomes = []
        for index in range(0, len(policy_actions)):
            total_return = sum(policy_imm_reward_list[index:]) + policy_reward
            all_returns.append(total_return)
            #policy_outcomes.append(total_return)
        policy_outcomes = len(policy_actions)*[shaped_policy_rewad]
        append_transitions(policy_states, policy_actions, policy_outcomes, all_transitions, use_symmetries, board_size,num_input_channels)
        opponent_outcomes = len(opponent_actions)*[shaped_policy_rewad]
        append_transitions(opponent_states, opponent_actions, opponent_outcomes, all_anti_transitions, use_symmetries, board_size,num_input_channels)
    elif opponent_reward == 1:
        opponent_outcomes = []

        for index in range(0, len(opponent_actions)):
            total_return = sum(opponent_imm_reward_list[index:]) + opponent_reward
            all_returns.append(total_return)
        opponent_outcomes = len(opponent_actions)*[shaped_opponent_reward]
            #opponent_outcomes.append(total_return)
        append_transitions(opponent_states, opponent_actions, opponent_outcomes, all_transitions, use_symmetries, board_size,num_input_channels)
        policy_outcomes = len(policy_actions)*[shaped_opponent_reward]
        append_transitions(policy_states, policy_actions, policy_outcomes, all_anti_transitions, use_symmetries, board_size,num_input_channels)


    if episode_num % num_episodes_per_update == 0:

        #mean_return = np.mean(all_returns)
        #std_return = np.std(all_returns)
        #for x in all_transitions: x[-1] = (x[-1]-mean_return)/std_return
        #for x in all_anti_transitions: x[-1] = (x[-1]-mean_return)/std_return

        #policy.update_model(all_transitions)
        #anti_policy.update_model(all_anti_transitions)

        opponent_pool.add_to_pool((policy.copy(), anti_policy.copy()))
        all_transitions = []
        all_anti_transitions = []
        opponent, anti_opponent = opponent_pool.sample_opponent()

    # analyze results
    if episode_num % 1000 == 0:
        policy.set_boltzmann_action(False)
        policy.set_epsilon(0.0)
        anti_policy.set_boltzmann_action(False)
        anti_policy.set_epsilon(0.0)

        results_self = {'won': 0, 'lost': 0, 'tied': 0}
        for p in opponent_pool._opponent_pool[0:-1]:
            p[0].set_epsilon(0)
            this_result = duel(board_size, policy, p[0],num_input_channels)
            results_self['won'] += this_result['won']
            results_self['lost'] += this_result['lost']

        winning_rate = (1.0*results_self['won']/(results_self['lost']+results_self['won']))
        print("%s,%s,%s" % (episode_num,winning_rate,len(opponent_pool._opponent_pool)-1))
        self_play_metric_file.write(str(episode_num)+"\t"+str(winning_rate)+"\n")

        if episode_num % 1000 ==0:
            self_play_metric_file.flush()
            opponents = [RandomPolicy(), Level1HeuristicPolicy(board_size), Level2HeuristicPolicy(board_size)]

            results = evaluate(policy, board_size, 1000, opponents,num_input_channels)

            combined_policy = PolicyForEvaluation(policy, anti_policy,anti_policy_rate)
            combined_results = evaluate(combined_policy, board_size, 1000, opponents,num_input_channels)

            print("%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s" % (episode_num, results[RandomPolicy.__name__]['won'],
                                                      results[Level1HeuristicPolicy.__name__]['won'],
                                                      results[Level2HeuristicPolicy.__name__]['won'],
                                                      combined_results[RandomPolicy.__name__]['won'],
                                                      combined_results[Level1HeuristicPolicy.__name__]['won'],
                                                      combined_results[Level2HeuristicPolicy.__name__]['won'],
                                                      results, len(unique_states_visited), epsilon, lr))
            random_winning_rate = results[RandomPolicy.__name__]['won']/(1000.0)
            l1_winning_rate = results[Level1HeuristicPolicy.__name__]['won']/1000.0
            l2_winning_rate = results[Level2HeuristicPolicy.__name__]['won']/1000.0

            random_winning_rate_combined = combined_results[RandomPolicy.__name__]['won'] / (1000.0)
            l1_winning_rate_combined = combined_results[Level1HeuristicPolicy.__name__]['won'] / 1000.0
            l2_winning_rate_combined = combined_results[Level2HeuristicPolicy.__name__]['won'] / 1000.0

            benchmark_play_metric_file.write(str(episode_num)+"\t"+str(random_winning_rate)+"\t"+str(l1_winning_rate)+"\t"+str(l2_winning_rate)+"\t"+str(random_winning_rate_combined)+"\t"+str(l1_winning_rate_combined)+"\t"+str(l2_winning_rate_combined)+"\n")
            WeightWriter.print_episode(results_file_base_path, experiment_name+"_weights_"+str(episode_num), policy.print_params)

            benchmark_play_metric_file.flush()


self_play_metric_file.close()
benchmark_play_metric_file.close()
WeightWriter.close()

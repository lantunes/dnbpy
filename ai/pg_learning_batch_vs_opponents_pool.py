import sys
sys.path.append("/Users/u6042446/Desktop/LuisRL/dnbpy/")
from ai import *
from dnbpy import *
from util.helper_functions import *
from util.file_helper import *
from util.reward_util import *
from util.rate_util import *
from util.evaluator import *
from util.opponent_pool_util import *

board_size = (3, 3)
num_episodes = 1500000
learning_rate_schedule = {0: 0.005, 1000000: 0.0005}
epsilon_rate_schedule = {300000:.2,500000:.1}
epsilon_supervised = 0.1
epsilon_self_play_max = 0.1
epsilon_self_play_min = 0.1

batch_size = 32
decay_speed = 1.0
opponent_pool_max_size = 50
num_episodes_per_policy_update = 100
num_episodes_per_opponent_cache = 1000
landa = .99
use_symmetries = True
try:
    base_path,min_eps,max_eps,exp_name = get_args()
except:
    min_eps = .1
    max_eps = .1
    exp_name = "test"
if not not min_eps: epsilon_self_play_min = min_eps
if not not max_eps: epsilon_self_play_max = max_eps
if not exp_name:
    print("Name your experiment by -N 'experiment_name'")
    sys.exit(1)


print("initializing for (%s, %s) game..." % (board_size[0], board_size[1]))
policy = PGPolicy3x3CNN(board_size, batch_size=batch_size, dropout_keep_prob=0.5)
#policy.restore_model("/Users/u6042446/Desktop/LuisRL/dnbpy/ai/Results/var_epsilon_min_01_max_03_model_399000/model.ckpt")
L2 = Level2HeuristicPolicy(board_size)
L1 = Level1HeuristicPolicy(board_size)
L0 = RandomPolicy()
reward_fn = DelayedBinaryReward()
opponent_pool = OpponentPool(max_size=opponent_pool_max_size)


#some further parameters
start_random_policy = 0
start_l1_policy = 10000
start_l2_policy = 110000
start_self_play = 300000
opponent_schedule = {start_random_policy: L0, start_l1_policy: L1, start_l2_policy: L2, start_self_play: 'policy'}

print_info(board_size=board_size, num_episodes=num_episodes, policy=policy, mode='watch opponents then self-play pool',
           reward=reward_fn, updates='offline', learning_rate_schedule=learning_rate_schedule, epsilon=epsilon_supervised,
           architecture=policy.get_architecture(), batch_size=batch_size)

unique_states_visited = set()
all_transitions = []

#Hacky way
metric_file = open(base_path+"/"+exp_name+"_metrics.txt","a")

for episode_num in range(1, num_episodes + 1):


    scheduled_opponent = gen_rate_step(episode_num, opponent_schedule)

    if scheduled_opponent == 'policy':
        #self-play
        #Learning-rate based on the exponential policy
        #lr = gen_rate_exponential(episode_num-start_self_play,.005,.00005,num_episodes-start_self_play)
        lr = gen_rate_step(episode_num, learning_rate_schedule)
        opponent = opponent_pool.sample_opponent()
        #eps = gen_rate_exponential((episode_num-start_self_play),
         #                          epsilon_self_play_max, epsilon_self_play_min, num_episodes-start_self_play, decay_speed)
        eps = gen_rate_step(episode_num,epsilon_rate_schedule)
    else:
        opponent = scheduled_opponent
        eps = epsilon_supervised
        lr = gen_rate_step(episode_num, learning_rate_schedule)

    policy.set_boltzmann_action(False)
    policy.set_epsilon(eps)
    policy.set_learning_rate(lr)

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
            edge = policy.select_edge(board_state)
            policy_actions.append(to_one_hot_action(board_state, edge))
            current_player, _ = game.select_edge(edge, current_player)
            unique_states_visited.add(as_string(game.get_board_state()))
        else:
            opponent_states.append(board_state)
            edge = opponent.select_edge(board_state)
            opponent_actions.append(to_one_hot_action(board_state, edge))
            current_player, _ = game.select_edge(edge, current_player)
            unique_states_visited.add(as_string(game.get_board_state()))
    policy_reward = reward_fn.compute_reward(game, 'policy', 'opponent')
    opponent_reward = reward_fn.compute_reward(game, 'opponent', 'policy')

    #Map to +1/-1
    policy_reward = np.sign(policy_reward-.1)
    opponent_reward = np.sign(opponent_reward-.1)

    # don't add transitions that have 0 reward as the gradient will be zero anyways
    all_returns = []

    if policy_reward == 1:
    #if True:
        decays = []
        policy_outcomes = len(policy_actions)*[policy_reward]
        #calcualte return
        [decays.append(landa**x) for x in range(0,len(policy_outcomes))]
        policy_return = (np.cumsum((np.array(policy_outcomes)*np.array(decays))[::-1]))[::-1]
        #policy_outcomes = list(policy_return)
        all_returns+= policy_outcomes
        append_transitions(policy_states, policy_actions, policy_outcomes, all_transitions, use_symmetries, board_size)

    elif opponent_reward == 1:
    #if True:
        decays = []
        opponent_outcomes = len(opponent_actions)*[opponent_reward]
        [decays.append(landa ** x) for x in range(0, len(opponent_outcomes))]
        opponent_return = (np.cumsum((np.array(opponent_outcomes) * np.array(decays))[::-1]))[::-1]
        #opponent_outcomes = list(opponent_return)
        all_returns+= opponent_outcomes
        append_transitions(opponent_states, opponent_actions, opponent_outcomes, all_transitions, use_symmetries, board_size)

    #sys.exit(1)
    #Calculate mean and standard deviation of policy values
    mean_rew = np.mean(all_returns)
    std_rew = np.std(all_returns)
    if episode_num % num_episodes_per_policy_update == 0:
        #sys.exit(0)
        #policy.update_model(all_transitions)
        policy.update_model(all_transitions)
        all_transitions = []
        all_returns = []

    if episode_num % num_episodes_per_opponent_cache == 0:
        opponent_pool.add_to_pool(policy.copy() if scheduled_opponent == 'policy' else opponent)

    #print(episode_num)
    # analyze results
    if episode_num % 1000 == 0 and episode_num>0:
        # play against opponents
        policy.set_boltzmann_action(False)
        policy.set_epsilon(0.0)
        opponents = [RandomPolicy(), Level1HeuristicPolicy(board_size), Level2HeuristicPolicy(board_size)]
        results = evaluate(policy, board_size, 1000, opponents)
        print("%s, %s, %s, %s, %s, %s, %s, %s" % (episode_num, results[RandomPolicy.__name__]['won'],
                                                  results[Level1HeuristicPolicy.__name__]['won'],
                                                  results[Level2HeuristicPolicy.__name__]['won'],
                                                  results, len(unique_states_visited), eps, lr))

        metric_file.write(str(episode_num)+"\t"+str(results[RandomPolicy.__name__]['won'])+"\t"+str(results[Level1HeuristicPolicy.__name__]['won'])
                          +"\t"+str(results[Level2HeuristicPolicy.__name__]['won'])+"\t"+str(len(unique_states_visited))+"\t"+str(eps)+"\t"+str(lr)+"\n")

        #WeightWriter.print_episode(base_path, episode_num, policy.print_params)
        #Create a folder for this model
        if episode_num % 10000 == 0 and episode_num > 0:
            model_folder_path = base_path+"/"+exp_name+"_model_"+str(episode_num)
            os.makedirs(model_folder_path)
            policy.store_model(model_folder_path,"model.ckpt")



metric_file.close()

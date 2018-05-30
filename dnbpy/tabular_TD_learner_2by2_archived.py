
from game import Game
import numpy as np
import random
from enum import Enum
import matplotlib.pyplot as plt
from utils import quadruple

NEW_NODE_SCORE = 0
EPSILON = .1
TEMPRETURE = 1
GAMMA = .99
INITIAL_LEARNING_RATE = 1
MIN_LEARNING_RATE = 1E-10
NUM_EPISODES = 1500000
NUM_ROWS = 2
NUM_COLUMNS = 2



class agent(Enum):
    RANDOM = 0
    TD = 1

class policy(Enum):
    GREEDY = 0
    EPSILON_GREEDY = 1

class node:
    def __init__(self,utility_val=None):
        #self._height = height
        self._utility_val = utility_val #the value of the current state (for TD learning)
        self._children = {} #list of children for this node==> key: edge-index, value: list of possiblities

    def set_utility_val(self,val): self._utility_val = val
    def get_utility_val(self): return (self._utility_val)


class Tri:
    def __init__(self,num_rows,num_cols):

        self._init_state_val = NEW_NODE_SCORE
        self._num_terminals = 0
        self._root = node(0) #Root has no utility value
        self._tree_height = ((2 * num_rows * num_cols) + num_rows + num_cols)
        #self.build_tri(self.tree_height,self._root)
        #self.build_tri(self.board_state_vector,self._root)


    def set_state_value(self,string_index,value):
        """
        :adding a string_index to the Tree
        :param string_index:
        :return:
        """
        if len(string_index)!=self._tree_height:
            raise Exception("Size of state vector is wrong")

        root = self._root
        node_buffer = [root]

        #while not not node_buffer:
        for idx,index in enumerate(string_index):
            current_node = node_buffer.pop()
            child = current_node._children.get(index)
            if not not child:
                node_buffer.append(current_node._children[index])
            else:
                child = node()
                current_node._children[index] = child
                node_buffer.append(child)
                self._num_terminals+=1

        terminal = node_buffer.pop()
        #if not terminal.get_utility_val():
        terminal.set_utility_val(value)

    def get_state_value(self,string_index):
        """
        Get a terminal according to the string_index
        :param string_index:
        :return:
        """

        if len(string_index)!=self._tree_height:
            raise Exception("Size of state vector is wrong")

        root = self._root
        node_buffer = [root]

        #while not not node_buffer:
        for idx,index in enumerate(string_index):
            current_node = node_buffer.pop()
            child = current_node._children.get(index)
            if not not child:
                node_buffer.append(current_node._children[index])
            else:
                child = node()
                current_node._children[index] = child
                node_buffer.append(child)
                self._num_terminals+=1

        terminal = node_buffer.pop()
        if not terminal.get_utility_val():terminal.set_utility_val(self._init_state_val)
        return (terminal)


def get_possible_states(current_state):
    """
    :param current_state: the current string_index as the state of the board
    :return: List of all possible states that the model can make a transition to
    """
    zero_index = []
    [zero_index.append(idx) for idx,val in enumerate(current_state) if val==0]

    possible_states = []
    for possible_index in zero_index:
        state = list(current_state)
        state[possible_index] = 1
        possible_states.append(state)
    return (possible_states)

def one_shot_play(current_state,game_tri,tempreture,agent_type,user_policy):
    """
    :param current_state: string_index corresponding to the current_state
    :param game_tri: Tri assigned to the game: terminals are game states
    :param tempreture: temperature for the Blotzeman distribution (this represents exploration)
    :param agent_type: Type of agent we play against
    :return: the next action
    """

    #Get all possible moves for the current_state
    possible_states = get_possible_states(current_state)
    #get all values
    state_vals = []
    probs = []
    for p in possible_states:
        terminal = game_tri.get_state_value(p)
        state_vals.append(terminal.get_utility_val())
        if agent_type==agent.TD:
            probs.append(np.exp(terminal.get_utility_val()/tempreture))
        elif agent_type==agent.RANDOM:
            probs.append(1.0)

    #Normalize probabilities
    probs = np.array(probs)/sum(probs)
    #draw an index using bionomial
    max_index = probs.argmax()
    if user_policy ==policy.EPSILON_GREEDY:
        random_sample_index = np.random.multinomial(1,probs).argmax()
    else:
        random_sample_index = probs.argmax()

    is_exploration = False
    if agent_type==agent.TD:
        if max_index!=random_sample_index:is_exploration = True

    next_move = possible_states[random_sample_index]
    return (np.nonzero(np.array(current_state)-next_move)[0][0],is_exploration)



def gen_learning_rate(iteration,l_max,l_min,N_max):
    """

    :param iteration: episode_index
    :param l_max: maximum value of the parameter
    :param l_min: minimum value of the parameter
    :param N_max: The maximum number of epsidosed after by which the value gets to min_value
    :return: The value expontially decaying over episodes
    """
    if iteration>N_max: return (l_min)
    alpha = 2*l_max
    beta = np.log((alpha/l_min-1))/N_max
    return (alpha/(1+np.exp(beta*iteration)))

def td1_learner(backups,reward,rewards,episode_index,training_quads_main,training_quads_opp):
    """

    :param backups:
    :param reward:
    :param rewards:
    :param episode_index:
    :param training_quads_main:
    :param training_quads_opp:
    :return:
    """

    #back-tracking
    learning_rate = gen_learning_rate(episode_index,.2,1E-4,1000000)
    #learning_rate = .1
    for index in np.arange(len(backups)-1,-1,-1):
    #for rec in training_quads_main:

        #TD-1 update

        #old_util = rec.get_s_t().get_utility_val()
        #if not rec.get_s_t_1():
         #   next_state_val = 0
        #else:
         #   next_state_val = rec.get_s_t_1().get_utility_val()
        #r = rec.get_reward()

        old_util = backups[index].get_utility_val()
        if index == len(backups)-1:
            next_state_val = 0
            r = reward+rewards[index]
        else:
            next_state_val = backups[index+1].get_utility_val()
            r = reward+rewards[index]
        #r = rewards[index+1]
        #TD-rule
        new_util = old_util + learning_rate*(r+GAMMA*next_state_val-old_util)
        #rec.get_s_t().set_utility_val(new_util)
        backups[index].set_utility_val(new_util)

def play_game(num_rows,num_columns,game_tri,tempreture,player_to_update,agent_type,user_policy):
    """
    Play one full episode here
    :param num_rows: number of rows in the board
    :param num_columns:  number of columns in the board
    :param game_tri: game-tri created so-far
    :param tempreture: exploration temperature
    :param player_to_update: the current player index
    :param agent_type: type of agent playing against
    :param user_policy: policy for playing (Greedy or Epsilon Greedy)
    :return:
    """
    # Launch the game-engine
    engine = Game((num_rows, num_columns), (0, 1))
    # Initialize board state
    current_state = [0] * ((2*NUM_ROWS*NUM_COLUMNS) + NUM_ROWS + NUM_COLUMNS)
    # form backups
    backups = []
    rewards = []
    training_quads_main = []
    training_quads_opp = []
    prev_state_main = None
    prev_state_opp = None
    while True:
        player_index = engine.get_current_player()
        # get edge index
        if player_index==player_to_update:
            line_index,is_exploration = one_shot_play(current_state, game_tri, tempreture, agent.TD,user_policy)
        else:
            #opponent (self-play or random-agent)
            line_index,is_exploration = one_shot_play(current_state, game_tri, tempreture, agent_type,user_policy)

        # change the game state
        player_index_out,immediate_reward = engine.select_edge(line_index, player_index)
        # check if the game is over

        # Update current state
        current_state = engine.get_board_state()
        if engine.is_finished(): break
        rewards.append(immediate_reward*.1)
        # store terminals for the target player
        if player_index==player_to_update:
            backups.append(game_tri.get_state_value(current_state))
            #This is the main player
            if not not prev_state_main:training_quads_main.append(quadruple(game_tri.get_state_value(prev_state_main),
                                                                            game_tri.get_state_value(current_state), immediate_reward, line_index))
            prev_state_main = list(current_state)
        else:
            if not not prev_state_opp:training_quads_opp.append(quadruple(game_tri.get_state_value(prev_state_opp), game_tri.get_state_value(current_state),                                                                              immediate_reward, line_index))
            prev_state_opp = list(current_state)

    # Get the final reward
    scores = []
    scores.append(engine.get_score(0))
    scores.append(engine.get_score(1))
    #print(scores)

    # Determine reward
    if scores[player_to_update] > scores[abs(1 - player_to_update)]:
        reward = 1
        #if agent_type==agent.RANDOM:
         #   print("Won:"+str(scores[player_to_update])+" to "+str(scores[abs(1 - player_to_update)]))
    else:
        reward = 0
        #if agent_type==agent.RANDOM:
         #   print("Lost:"+str(scores[player_to_update])+" to "+str(scores[abs(1 - player_to_update)]))
    rewards.append(reward)

    # attach last quad
    if player_index==player_to_update:
        training_quads_main.append(quadruple(game_tri.get_state_value(prev_state_main), game_tri.get_state_value(current_state), reward, line_index))
        training_quads_opp.append(quadruple(game_tri.get_state_value(prev_state_opp), None, reward, line_index))
    else:
        training_quads_main.append(quadruple(game_tri.get_state_value(prev_state_main), None, reward, line_index))
        training_quads_opp.append(quadruple(game_tri.get_state_value(prev_state_opp), game_tri.get_state_value(current_state), reward, line_index))

    return (reward,backups,rewards,training_quads_main,training_quads_opp)

if __name__=="__main__":

    #Performance metrics:
    #1) total reward against random player
    #2) some state values evolution

    #sample_state1 = [1,0,1,1,1,0,0,0,0,0,0,0]
    #sample_state2 = [1,0,1,1,0,1,0,0,0,0,0,0]

    #Initialize game_tri

    game_tri = Tri(NUM_ROWS,NUM_COLUMNS)
    #Initial state
    rewards = []
    reward_saver = []
    state_saver1 = []
    state_saver2 = []

    player_to_update = 0
    tempreture = 1
    #Main loop over all the episodes
    for episode_index in range(0,NUM_EPISODES):
        tempreture = gen_learning_rate(episode_index,1,.05,1000000) #slighty decay the tempreture
        #tempreture = 1
        #if episode_index%50000==0:
         #   tempreture = max(.1,tempreture-.1)
        #randomly choose one player
        if random.random()<.5:player_to_update=1-player_to_update
        #get reward and backups
        reward,backups,rewards,training_quads_main,training_quads_opp = play_game(NUM_ROWS,NUM_COLUMNS,game_tri,tempreture,player_to_update,agent.TD, policy.EPSILON_GREEDY)

        #TD-1 learner
        td1_learner(backups,reward,rewards,episode_index,training_quads_main,training_quads_opp)
        if episode_index%1000==0:
            #state_saver1.append(game_tri.get_state_value(sample_state1).get_utility_val())
            #state_saver2.append(game_tri.get_state_value(sample_state2).get_utility_val())
            #play against a random-player
            total_reward = 0
            test_player_index = 0
            for test_episode in range(0,1000):
                if random.random() < .5: test_player_index = 1 - test_player_index
                reward, backups,rewards,training_quads_main,training_quads_opp = play_game(NUM_ROWS, NUM_COLUMNS, game_tri, .01, test_player_index, agent.RANDOM, policy.EPSILON_GREEDY)
                if np.sign(reward)>0:
                    total_reward+= 1
            reward_saver.append(total_reward)
            if episode_index%10000==0: print(reward_saver[-1])
            #print(episode_index)

        #if episode_index%10000==0: print(reward_saver[-1])

    #plot some results
    plt1, = plt.plot(range(1,len(reward_saver)+1),(np.array(reward_saver)/1000.0)*100,"k--",linewidth = 2)
    plt.xlabel(r"Training Iteration$\times$1000",size=20)
    plt.ylabel("Winning Rate (%)", size=20)
    plt.grid(True)
    plt.show()

    plt2, = plt.plot(range(1,len(state_saver1)+1),state_saver1,"bo-",linewidth = 2)
    plt3, = plt.plot(range(1, len(state_saver2) + 1), state_saver2, "rd-", linewidth=2)
    plt.xlabel(r"Training Iteration$\times$1000", size=20)
    plt.ylabel("State Value", size=20)
    plt.legend([plt2,plt3],["First State","Second State"])
    plt.grid(True)
    plt.show()








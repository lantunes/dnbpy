


NEW_NODE_SCORE = 1E-6
EPSILON = .1
TEMPRETURE = 10
GAMMA = 1
INITIAL_LEARNING_RATE = 1E-3
MIN_LEARNING_RATE = 1E-6
import numpy as np
import random
NUM_EPISODES = 50000

from dnbpy import GameEngine
import numpy as np
import random
from enum import Enum
import matplotlib.pyplot as plt

NEW_NODE_SCORE = 1E-6
EPSILON = .1
TEMPRETURE = 5
GAMMA = .99
INITIAL_LEARNING_RATE = 1
MIN_LEARNING_RATE = 1E-6
NUM_EPISODES = 100000
NUM_ROWS = 2
NUM_COLUMNS = 2



class agent(Enum):
    RANDOM = 0
    TD = 1

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

        terminal = node_buffer.pop()
        if not terminal.get_utility_val():terminal.set_utility_val(self._init_state_val)
        return (terminal)


def get_possible_states(current_state):
    zero_index = []
    [zero_index.append(idx) for idx,val in enumerate(current_state) if val==0]

    possible_states = []
    for possible_index in zero_index:
        state = list(current_state)
        state[possible_index] = 1
        possible_states.append(state)
    return (possible_states)


def one_shot_play(current_state,game_tri,tempreture,agent_type):
    """
    :param current_state:
    :param game_tri:
    :param tempreture:
    :param agent_type:
    :return:
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
            probs.append(1)

        probs.append(np.exp(terminal.get_utility_val()/tempreture))

    #Normalize state values
    state_vals = np.array(state_vals)/sum(state_vals)
    #probs = np.array(probs)/(len(probs)-1)
    #Use Epsilon-greedy to choose a new state
    #max_val = max(state_vals)
    #for idx, val in enumerate(state_vals):
     #   if val==max_val:
      #      probs[idx] = 1-EPSILON
       # else:
        #    probs[idx] = EPSILON

    #Normalize probabilities
    probs = np.array(probs)/sum(probs)
    #draw an index using bionomial
    random_sample_index = np.random.multinomial(1,probs).argmax()
    next_move = possible_states[random_sample_index]
    return (np.nonzero(np.array(current_state)-next_move)[0][0])



def gen_learning_rate(iteration,l_max,l_min,N_max):
    if iteration>N_max: return (l_min)
    alpha = 2*l_max
    beta = np.log((alpha/l_min-1))/N_max
    return (alpha/(1+np.exp(beta*iteration)))

def td1_learner(backups,reward,episode_index):
    """

    :param backups:
    :param reward:
    :param game_tri:
    :return:
    """

    #back-tracking
    learning_rate = gen_learning_rate(episode_index,INITIAL_LEARNING_RATE,MIN_LEARNING_RATE,NUM_EPISODES)
    for index in np.arange(len(backups)-1,-1,-1):

        #TD-1 update
        old_util = backups[index].get_utility_val()
        if index == len(backups)-1:
            next_state_val = 0
            r = reward
        else:
            next_state_val = backups[index+1].get_utility_val()
            r = 0
        #TD-rule
        new_util = old_util + learning_rate*(r+GAMMA*next_state_val-old_util)
        backups[index].set_utility_val(new_util)


def play_game(num_rows,num_columns,game_tri,tempreture,player_to_update,agent_type):
    # Launch the game-engine
    engine = GameEngine((num_rows, num_columns), (0, 1))
    # Initialize board state
    current_state = [0] * 12
    # form backups
    backups = []
    while True:
        player_index = engine.get_current_player()
        # get edge index
        if player_index==player_to_update:
            line_index = one_shot_play(current_state, game_tri, tempreture, agent.TD)
        else:
            #opponent (self-play or random-agent)
            line_index = one_shot_play(current_state, game_tri, tempreture, agent_type)

        # change the game state
        engine.select_edge(line_index, player_index)
        # check if the game is over
        if engine.is_game_finished(): break
        # Update current state
        current_state = engine.get_board_state()
        # store terminals for the target player
        if player_index == player_to_update: backups.append(game_tri.get_state_value(current_state))

    # Get the final reward
    scores = []
    scores.append(engine.get_score(0))
    scores.append(engine.get_score(1))

    # Determine reward
    if scores[player_to_update] > scores[abs(1 - player_to_update)]:
        reward = 1
    else:
        reward = 0

    return (reward,backups)

if __name__=="__main__":

    #Performance metrics:
    #1) total reward against random player
    #2) some state values evolution

    sample_state1 = [1,0,1,1,0,0,0,0,0,0,0,0]
    sample_state2 = [1,0,1,1,0,1,0,0,0,0,0,0]

    #Initialize game_tri

    game_tri = Tri(NUM_ROWS,NUM_COLUMNS)
    #Initial state
    rewards = []
    reward_saver = []
    state_saver1 = []
    state_saver2 = []

    player_to_update = 0
    #Main loop over all the episodes
    for episode_index in range(0,NUM_EPISODES):
        tempreture = gen_learning_rate(episode_index,TEMPRETURE,.1,NUM_EPISODES) #slighty decay the tempreture
        #randomly choose one player
        if random.random()<.5:player_to_update=1-player_to_update
        #get reward and backups
        reward,backups = play_game(NUM_ROWS,NUM_COLUMNS,game_tri,tempreture,player_to_update,agent.TD)
        #TD-1 learner
        td1_learner(backups,reward,episode_index)
        if episode_index%500==0:
            state_saver1.append(game_tri.get_state_value(sample_state1).get_utility_val())
            state_saver2.append(game_tri.get_state_value(sample_state2).get_utility_val())
            #play against a random-player
            total_reward = 0
            test_player_index = 0
            for test_episode in range(0,500):
                if random.random() < .5: test_player_index = 1 - test_player_index
                reward, backups = play_game(NUM_ROWS, NUM_COLUMNS, game_tri, .01, test_player_index, agent.RANDOM)
                total_reward+= reward
            reward_saver.append(total_reward)
            print(episode_index)

    #plot some results
    plt1, = plt.plot(range(1,len(reward_saver)+1),reward_saver,"k--",linewidth = 2)
    plt.xlabel(r"Training Iteration$\times$100",size=20)
    plt.ylabel("Sum Reward", size=20)
    plt.grid(True)
    plt.show()

    plt2, = plt.plot(range(1,len(state_saver1)+1),state_saver1,"bo-",linewidth = 2)
    plt3, = plt.plot(range(1, len(state_saver2) + 1), state_saver2, "rd-", linewidth=2)
    plt.xlabel(r"Training Iteration$\times$100", size=20)
    plt.ylabel("State Value", size=20)
    plt.legend([plt2,plt3],["First State","Second State"])
    plt.grid(True)
    plt.show()

=======
if __name__=="__main__":

    #Initialize game_tri
    game_tri = Tri(2,2)
>>>>>>> Stashed changes

    #Initial state
    rewards = []
    reward_saver = []

    player_to_update = 0
    for episode_index in range(0,NUM_EPISODES):
        tempreture = gen_learning_rate(episode_index,TEMPRETURE,.01,NUM_EPISODES)
        #choose the player for training
        #if episode_index%1000==0 and episode_index>0:
         #   player_to_update = 1-player_to_update

        if random.random()<.5:
            player_to_update=0
        else:
            player_to_update = 1

        #print(player_to_update)

        # Launch the game-engine
        engine = GameEngine((2, 2), (0, 1))
        current_state  = [0]*12
        backups = {}

        game_condition = True
        while game_condition:
            player_index = engine.get_current_player()
            line_index = one_shot_play(current_state,game_tri,tempreture)
            engine.select_edge(line_index,player_index)
            if engine.is_game_finished():
                game_condition = False
                break
            current_state = engine.get_board_state()

            if not backups.has_key(player_index): backups[player_index] = []
            backups[player_index].append(game_tri.get_state_value(current_state)) #store all the terminals


        #Get the final reward
        scores = []
        scores.append(engine.get_score(0))
        scores.append(engine.get_score(1))

        #Determine reward
        if scores[player_to_update]>scores[abs(1-player_to_update)]:
            reward = 1
        else:
            reward = 0

        if episode_index%500==0 and episode_index>0:
            reward_saver.append(sum(rewards))
            print(episode_index,tempreture,sum(rewards))
            rewards = []
            #print(reward_saver)
        else:
            rewards.append(reward)


        #TD-1 learner
        td1_learner(backups[player_to_update],reward,episode_index)
        #td1_learner(backups[1-player_to_update], 1-reward, episode_index)






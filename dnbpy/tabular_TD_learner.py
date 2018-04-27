NEW_NODE_SCORE = .1
EPSILON = .1
import numpy as np
from dnbpy import GameEngine


class node:
    def __init__(self,utility_val=None):
        #self._height = height
        self._utility_val = utility_val #the value of the current state (for TD learning)
        self._children = {} #list of children for this node==> key: edge-index, value: list of possiblities

    def set_utility_val(self,val): self._utility_val = val
    def get_utility_val(self): return (self._utility_val)


class Tri:
    def __init__(self,num_rows,num_cols):

        self._init_state_val = .1
        self._num_terminals = 0
        self._root = node(0) #Root has no utility value
        self._tree_height = ((2 * num_rows * num_cols) + num_rows + num_cols)
        #self.build_tri(self.tree_height,self._root)
        #self.build_tri(self.board_state_vector,self._root)

    """
    def build_tri(self,tree_height,parent):
        
        #parent = self._root #start from the root

        #Count number of zero entries in board_state_vector for initial probability assignment

        #Breadth-first filling
        #board_state_buffer = [board_state_vector]
        node_buffer = [parent]
        while not not node_buffer:
            print(len(node_buffer))
            current_parent = node_buffer.pop(0)
            #node_buffer = node_buffer[1:]
            #current_parent = node_buffer.pop()
            if current_parent._height==tree_height:
                #this is terminal
                self._num_terminals+=1
                #print(self._num_terminals)
                continue
            #Form initial_prob for all possible movements
            left_child = node(current_parent._height+1)
            right_child = node(current_parent._height+1)
            current_parent._children[0] = left_child
            current_parent._children[1] = right_child
            node_buffer.append(left_child)
            node_buffer.append(right_child)
    """


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

def one_shot_play(current_state,game_tri):
    """
    :Get the next index for play
    :param current_state:
    :param game_tri:
    :return:
    """
    possible_states = get_possible_states(current_state)
    #get all values
    state_vals = []
    probs = []
    for p in possible_states:
        terminal = game_tri.get_state_value(p)
        state_vals.append(terminal.get_utility_val())
        probs.append(0)

    #Normalize state values
    state_vals = np.array(state_vals)/sum(state_vals)
    #probs = np.array(probs)/(len(probs)-1)
    #Use Epsilon-greedy to choose a new state
    max_val = max(state_vals)
    for idx, val in enumerate(state_vals):
        if val==max_val:
            probs[idx] = 1-EPSILON
        else:
            probs[idx] = EPSILON

    probs = np.array(probs)/sum(probs)
    #draw an index using bionomial
    random_sample_index = np.random.multinomial(1,probs).argmax()
    next_move = possible_states[random_sample_index]
    #Find nonzero index

    return (np.nonzero(np.array(current_state)-next_move)[0][0])

if __name__=="__main__":
    game_tri = Tri(2,2)
    state = [0]*12
    state1 = [0] * 12
    state2 = [0] * 12
    state[0] = 1
    state1[0] =1
    state1[5] = 1
    state2[-1]= 1

    engine = GameEngine((2, 2), (0, 1))
    current_state  = [0]*12

    while not engine.is_game_finished():
        player_index = engine.get_current_player()
        line_index = one_shot_play(current_state,game_tri)
        engine.select_edge(line_index,player_index)
        current_state = engine.get_board_state()







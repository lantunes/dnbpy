NEW_NODE_SCORE = .1
import numpy as np

class node:
    def __init__(self,utility_val):
        self._utility_val = utility_val #the value of the current state (for TD learning)
        self._children = {} #list of children for this node==> key: edge-index, value: list of possiblities


class Tri:
    def __init__(self,num_rows,num_cols):

        self._root = node(-1) #Root has no utility value
        self.board_state_vector = [0] * ((2 * num_rows * num_cols) + num_rows + num_cols)
        self.build_tri(self.board_state_vector,self._root)
        #self.build_tri(self.board_state_vector,self._root)

    def build_tri(self,board_state_vector,parent):
        """

        :param num_rows:
        :param num_cols:
        :return:
        """

        #parent = self._root #start from the root

        #Count number of zero entries in board_state_vector for initial probability assignment

        #Breadth-first filling
        board_state_buffer = [board_state_vector]
        node_buffer = [parent]

        while (not not board_state_buffer):
            print(len(board_state_buffer))
            current_state_vector = board_state_buffer[0]
            current_parent = node_buffer[0]
            board_state_buffer = board_state_buffer[1:]
            node_buffer = node_buffer[1:]
            #Form initial_prob for all possible movements
            init_prob = 1.0/(len(current_state_vector)-np.count_nonzero(current_state_vector))
            #print(init_prob)
            #Iterate over possible moves and add them to the list
            for string_index,string_value in enumerate(current_state_vector):
                if string_value==0:
                    #This is a valid move
                    new_child = node(init_prob) #New child with initial probability
                    current_parent._children[string_index] = new_child
                    new_state_vector = list(current_state_vector)
                    new_state_vector[string_index]+=1
                    #add both to the buffer
                    board_state_buffer.append(new_state_vector)
                    node_buffer.append(new_child)


if __name__=="__main__":
    game_tri = Tri(2,2)



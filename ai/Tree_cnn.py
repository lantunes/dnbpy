import tensorflow as tf
import sys
sys.path.append("/Users/u6042446/Desktop/LuisRL/dnbpy/")
from dnbpy import *
import numpy as np


class Tree_cnn:
    def __init__(self,dim_board_x,dim_board_y):
        self.dim_board_x = dim_board_x
        self.dim_board_y = dim_board_y



class Tree_node:
    def __init__(self,value):
        self.value = value
        self.sub_board_top_left
        self.sub_board_top_right
        self.sub_board_bottom_left
        self.sub_board_bottom_right

def gen_sub_boards(save_all_ranges):
    index = 0
    while True:
        x_range = save_all_ranges[index][0]
        y_range = save_all_ranges[index][1]
        left_x = x_range[0];right_x = x_range[1]
        up_y = y_range[0]; down_y = y_range[1]
        if abs(right_x - left_x) == 2 and abs(up_y - down_y) == 2: break
        save_all_ranges = save_all_ranges[1:]

        top_left = [[left_x,right_x-1],[up_y,down_y-1]]
        top_right = [[left_x+1,right_x],[up_y,down_y-1]]
        bot_left = [[left_x,right_x-1],[up_y+1,down_y]]
        bot_right = [[left_x+1,right_x],[up_y+1,down_y]]
        save_all_ranges = save_all_ranges + [top_left] + [top_right] + [bot_left] + [bot_right]
    return (save_all_ranges)

def get_sub_edge_matrix(main_edge_matrix,sub_slice):
    x_range = sub_slice[0]
    y_range = sub_slice[1]

    x_start = x_range[0]*2
    x_end = x_range[1]*2
    y_start = y_range[0]*2
    y_end = y_range[1]*2
    sub_edge = np.array(main_edge_matrix[x_start:x_end+1,y_start:y_end+1])
    return (sub_edge)


if __name__=="__main__":
    board_size = (4,4)
    edge_matrix = init_edge_matrix((4,4))
    save_all_ranges = [[[0,5],[0,5]]]
    save_all_ranges = gen_sub_boards(save_all_ranges)

    players = [0, 1]
    game = Game(board_size, players)
    current_player = game.get_current_player()

    p0_actions = []
    p1_actions = []
    p0_states = []
    p1_states = []

    while not game.is_finished():
        board_state = game.get_board_state()
        edge_matrix = convert_board_state_to_edge_matrix(board_size, board_state)
        print(edge_matrix)
        #Temp
        legal_indices = []
        [legal_indices.append(x) for x,y in enumerate(board_state) if y==0]
        prob = np.ones([len(legal_indices)])/(1.0*len(legal_indices))
        edge = np.random.choice(legal_indices,
                                          p=np.array(prob))
        if current_player == 0:
            p0_states.append(board_state)

            #p0_actions.append(to_one_hot_action(board_state, edge))
            current_player, _ = game.select_edge(edge, 0)
            #unique_states_visited.add(as_string(game.get_board_state()))
        else:
            p1_states.append(board_state)
            #edge = policy.select_edge(board_state)
            #p1_actions.append(to_one_hot_action(board_state, edge))
            current_player, _ = game.select_edge(edge, 1)
            #unique_states_visited.add(as_string(game.get_board_state()))

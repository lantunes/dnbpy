from __future__ import division, print_function, absolute_import

from enum import Enum
from params import mlp_params
import pandas as pd
import csv
import tensorflow as tf
import numpy as np
import random
from params import Type
from utils import *
from dnbpy import GameEngine


SEED = 1000
random.seed(SEED)
np.random.seed(SEED)

class agent(Enum):
    RANDOM = 0
    TD = 1
    MLP = 2

class mlp_policy_network:
    def __init__(self,mlp_params):
        self._params = mlp_params
        # tf Graph input
        self._is_training = True
        self.construct_mlp_model()
        self.sess = tf.Session(graph=self.graph)

    def _init_weights_and_biases(self,mlp_params):
        # Initialize weights and biases
        layers = mlp_params.layers
        #conv_layers = layers[0:-1]
        #out_layer = layers[-1]

        self.mlp_weights = []
        self.mlp_biases = []
        self.out_weights = []
        self.out_biases = []

        prev_filter_size = 1
        for layer in layers:
            self.mlp_weights.append(tf.Variable(tf.random_normal(
                [layer.dim_input,layer.num_hidden_units])))
            self.mlp_biases.append(tf.Variable(tf.random_normal([layer.num_hidden_units])))

        #self.out_weights.append(tf.Variable(tf.random_normal([out_layer.dim_input, out_layer.dim_output])))
        #self.out_biases.append(tf.random_normal([out_layer.dim_output]))
        #self.out_weights.append(tf.Variable(tf.random_normal([out_layer.dim_output, conv_params.num_outputs])))
        #self.out_biases.append(tf.random_normal([conv_params.num_outputs]))

    def multilayer_perceptron_prediction(self,x):
        """
        :param x: input string-index
        :return: feature-vector for the output Softmax layer
        """
        #Loop over all the layers of the network
        mlp = x
        for layer_index in range(0,len(self.mlp_weights)):
            mlp_out = tf.add(tf.matmul(mlp,self.mlp_weights[layer_index]),self.mlp_biases[layer_index]) #output of this layer
            #check for nonlinearity
            if self._params.layers[layer_index].nonlinearity_type==Type.Relu:
                mlp = tf.nn.relu(mlp_out)
            elif self._params.layers[layer_index].nonlinearity_type==Type.Sigmoidal:
                mlp = tf.nn.sigmoid(mlp_out)
            elif self._params.layers[layer_index].nonlinearity_type==Type.Tanh:
                mlp = tf.nn.tanh(mlp_out)

        #Illegal moves should be strictly penalized: this ensures that the probability for illegal moves is zero
        #if not self._is_training:mlp_out = tf.add(mlp_out,tf.log(1-x))
        #mlp_out = tf.add(mlp_out, x*1000000000)

        return (mlp_out)


    def softmax_with_valid_moves(self,logits,string_index):
        """
        :param logits: feature-vector for the softmax-layer
        :param string_index: raw-string-index
        :return: softmax probability
        """

        exp_logits_with_valid_moves = tf.exp(logits)*(1-string_index)
        #normalization and softmax output
        softmax_out = exp_logits_with_valid_moves/tf.reduce_sum(exp_logits_with_valid_moves, reduction_indices=1)
        return (softmax_out)



    def construct_mlp_model(self):
        """
        :this initializes TensorGraph and all the required parameters
        :return:
        """
        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device('/cpu:0'):
            tf.set_random_seed(SEED)
            # Construct model
            self._input = tf.placeholder(tf.float32, [None, self._params.num_input])
            self._output = tf.placeholder(tf.float32, [None, self._params.num_outputs])
            self._reward = tf.placeholder(tf.float32,[None,1])
            self._learning_rate = tf.placeholder(tf.float32, shape=[None,1])

            self._keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)
            self._init_weights_and_biases(self._params) #initialize the graph and required parameters
            self.logits = self.multilayer_perceptron_prediction(self._input) #Apply MLP map to the input and get output values
            self.prediction_supervised = tf.nn.softmax(self.logits)
            self.prediction = self.softmax_with_valid_moves(self.logits,self._input)
            self.loss_for_pg = self.policy_loss(self.prediction,self._output,self._reward)
            self.optimizer_pg = tf.train.GradientDescentOptimizer(learning_rate=self._params.learning_rate)
            self.train_pg = self.optimizer_pg.minimize(self.loss_for_pg)

            # Define loss and optimizer
            self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self._output))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self._params.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss_op)
            self.correct_pred = tf.equal(tf.argmax(self.prediction_supervised, 1), tf.argmax(self._output, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
            # Evaluate model (write proper optimization)
            #self.correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self._output, 1))
            #self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
            self.init_op = tf.initialize_all_variables()

    def policy_loss(self,prediction,action,reward):
        """
        :param prediction: vector of softmax predictions
        :param action: chosen move by the player
        :param reward: return at the end of the game
        :return:
        """
        loss = -tf.log(tf.reduce_sum(prediction*action,reduction_indices=1))*reward
        return (loss)

    def get_weight(self):return (self.mlp_weights[-1])
    def display_weight(self):
        session = self.sess
        session.run(self.init_op)
        out = session.run(self.get_weight())
        return (out[0][0])


    def mlp_play(self,string_index):
        """
        :TASK generartes a probability vector for given string_index
        :param string_index: state of the game
        :return: probability for any of edges (illegal moves will have zero probabilities)
        """
        session = self.sess
        session.run(self.init_op)
        #feed_dict = {self._input: string_index}

        feed_dict = {self._input: string_index}
        out_probability = session.run(self.prediction, feed_dict=feed_dict)
        return (out_probability)

    def mlp_pg_train(self,states,returns,learning_rate):
        """
        :param states: list of all visited states
        :param returns: returned reward
        :param learning_rate: adaptive learning rate
        :return:
        """

        session = self.sess
        session.run(self.init_op)
        for idx,record in enumerate(states):
            current_state = np.array(record[0]).reshape([1,self._params.num_input])
            action = np.zeros([1,self._params.num_outputs])
            action[0,record[1]] = 1
            #print(action)
            feed_dict = {self._input: current_state,self._output: action,self._reward: np.array([returns]).reshape([1,1]),self._learning_rate:learning_rate}
            op = session.run(self.train_pg,feed_dict=feed_dict) #training using a simple PG rule

    def train_mlp_policy(self,data,do_evaluation):
        """
        :param data:
        :return:
        """
        session = self.sess
        session.run(self.init_op)

        for epoch in range(0,self._params.num_epochs):
            #shuffle data-frame
            print("Shuffling data-frame for the current epoch...")
            #shuffeled_data = data.reindex(np.random.permutation(data.index))
            shuffeled_data = data
            reset_data_indexes()

            index_list = list(set(shuffeled_data['index']))
            #Create train/test folds
            print("Folding the data into train and test sets")
            folds = n_fold_cross_validation(shuffeled_data, self._params.num_folds, index_list)
            if do_evaluation:
                fold_index = 0
            else:
                fold_index = -1

            #for fold_index, fold in enumerate(folds):
                #print("Fold:"+str(fold_index)+"....")
                #create train and test data-frames
            train_df, test_df = generate_train_validation_form_fold(fold_index, folds)
            #extract train/test states and labels
            train_states = list(train_df['state'])
            train_labels = list(train_df['move'])
            if do_evaluation:
                test_states = list(test_df['state'])
                test_labels = list(test_df['move'])

                #generate array of test-data
                test_state_batch,test_label_batch = generate_for_mlp_test(test_states, test_labels, self._params.dim_state)

            #Calculate total number of batches
            num_of_batches = int(len(train_states)/(self._params.batch_size))
            print("Number of batches is:"+str(num_of_batches))
            #Average training-loss
            average_loss = 0
            for batch_index in range(0,num_of_batches):
                #Generate a batch of train and test data
                train_state_batch, train_label_batch = generate_batch_for_mlp_training(train_states, train_labels, self._params.batch_size, self._params.dim_state)
                #Train using this batch (using cross-entropy as the loss function)
                feed_dict = {self._input: train_state_batch,self._output: train_label_batch}
                op, l = session.run([self.train_op,self.loss_op], feed_dict=feed_dict)

                average_loss += l
                if do_evaluation:
                    if batch_index % 500 == 0:
                        if batch_index > 0:
                            #average_loss = average_loss / 200
                            # The average loss is an estimate of the loss over the last 2000 batches.

                            #Also, run a test and calculate the performance on the test-set
                            feed_dict = {self._input: test_state_batch, self._output: test_label_batch}
                            #test_loss = session.run(self.loss_op,feed_dict=feed_dict)
                            test_accuracy = session.run(self.accuracy,feed_dict=feed_dict)
                            print('Average train/test-loss at step %d: %f,%f' % (batch_index, average_loss/500.0,test_accuracy))
                            average_loss = 0



def get_random_index(current_state):
    """
    :param current_state: the current state of the board
    :return: a randomlu selected edge from the board
    """
    valid_indexes = []
    probs = []
    for idx,c in enumerate(current_state):
        if c==0:
            valid_indexes.append(idx)
            probs.append(1)

    line_index = np.random.multinomial(1, np.array(probs)/sum(probs)).argmax()
    return (valid_indexes[line_index])

def one_shot_board_play(current_state,mlp_policy,tempreture,agent_type):
    """
    :param current_state:
    :param game_tri:
    :param tempreture:
    :param agent_type:
    :return:
    """

    if agent_type==agent.MLP:
        temp_probs = mlp_policy.mlp_play(np.array(current_state).reshape([1,len(current_state)]))[0] #policy network
        probs = []
        for x in temp_probs:probs.append(x/tempreture)
        probs = np.array(probs)/np.sum(probs)
        random_sample_index = np.random.multinomial(1, probs).argmax()
        next_move = random_sample_index
        #print(temp_probs,next_move)
    elif agent_type==agent.RANDOM:
        #Uniformly sample from valid moves
        next_move = get_random_index(current_state)
        pass

    return (next_move)

def gen_learning_rate(iteration,l_max,l_min,N_max):
    """

    :param iteration: episode_index
    :param l_max: maximum learning-rate
    :param l_min: minimum learning-rate
    :param N_max: maximum number of episodes
    :return:
    """
    if iteration>N_max: return (l_min)
    alpha = 2*l_max
    beta = np.log((alpha/l_min-1))/N_max
    return (alpha/(1+np.exp(beta*iteration)))

def play_game(num_rows,num_columns,mlp_policy,tempreture,player_to_update,agent_type):
    """
    :param num_rows:  number of rows in the board
    :param num_columns: number of columns in the board
    :param tempreture: current tempreture for Boltzeman distribution
    :param player_to_update: the player-index
    :param agent_type: type of agent (MLP or RANDOM)
    :return:
    """
    # Launch the game-engine
    engine = GameEngine((num_rows, num_columns), (0, 1))
    # Initialize board state
    current_state = engine._board_state

    # form backups
    backups = {}
    sofar = []
    while True:
        player_index = engine.get_current_player()

        # get edge index
        if player_index == player_to_update:
            line_index = one_shot_board_play(current_state, mlp_policy, tempreture, agent.MLP)
        else:
            line_index = one_shot_board_play(current_state, mlp_policy, tempreture, agent_type)

        # store visited states + actions for policy update

        if not player_index in backups.keys():backups[player_index] = []
        backups[player_index].append([list(current_state),line_index])
        #backups[player_index].append([current_state, line_index])  # store all the visited states and actions as well

        # change the game state
        engine.select_edge(line_index, player_index)
        # check if the game is over
        if engine.is_game_finished(): break
        # Update current state
        current_state = engine.get_board_state()
    # Get the final reward
    scores = []
    scores.append(engine.get_score(0))
    scores.append(engine.get_score(1))
    #print(scores)

    # Determine reward
    if scores[player_to_update] > scores[abs(1 - player_to_update)]:
        reward = 1
    else:
        reward = 0
    return (reward,backups)


def supervised_initialization(mlp_policy):
    """

    :param mlp_policy:
    :return:
    """

    data_file = "data/dnb_moves_indexed_dedup.txt"
    data = pd.read_csv(data_file, sep=",", quoting=csv.QUOTE_ALL)
    mlp_policy.train_mlp_policy(data,False)


if __name__=="__main__":

    #Construct mlp_network and policy object
    mlp_object = mlp_params(3, 3, [])
    mlp_policy = mlp_policy_network(mlp_object)
    #supervised_initialization(mlp_policy)
    #play game with the new network
    #test_player_index = 0
    #total_reward = 0
    #for test_episode in range(0, 10000):
     #   if random.random() < .5: test_player_index = 1 - test_player_index
      #  reward, backups = play_game(3, 3, mlp_policy, .01, test_player_index, agent.RANDOM)
      #  total_reward += reward
    #print(total_reward)


    # Initial state
    rewards = []
    reward_saver = []
    player_to_update = 0
    N_max = 100000

    #Big loop over all the episodes
    for episode_index in range(0, N_max):
        tempreture = gen_learning_rate(episode_index, 10, .01, N_max)
        learning_rate = gen_learning_rate(episode_index,1E-3,1E-8,N_max)
        #print(tempreture,learning_rate)
        #Choose the player_index first
        if random.random() < .5:
            player_to_update = 0
        else:
            player_to_update = 1

        #play game and generate reward and list of visited states
        reward,backups = play_game(3,3,mlp_policy,tempreture,player_to_update,agent.MLP)
        #Update weights of the MLP with policy-gradiant
        mlp_policy.mlp_pg_train(backups[player_to_update],reward,np.array([learning_rate]).reshape([1,1]))

        if episode_index%1000==0:
            #state_saver1.append(game_tri.get_state_value(sample_state1).get_utility_val())
            #state_saver2.append(game_tri.get_state_value(sample_state2).get_utility_val())
            #play against a random-player
            print(episode_index)
            total_reward = 0
            test_player_index = 0
            for test_episode in range(0,500):
                if random.random() < .5: test_player_index = 1 - test_player_index
                reward, backups = play_game(3, 3, mlp_policy,.005, test_player_index, agent.RANDOM)
                total_reward+= reward
            print(total_reward)
            reward_saver.append(total_reward)


    #print(mlp_policy.display_weight())
    #input = [0]*24
    #input[0] = 1
    #input[10] = 1
    #print(mlp_policy.mlp_play(np.array(input).reshape([1,24])))

    #sys.exit(1)
    #Read training data and extract states and labels
    #data_file = "data/dnb_moves_indexed_dedup.txt"
    #data = pd.read_csv(data_file, sep=",", quoting=csv.QUOTE_ALL)
    #mlp_policy.train_mlp_policy(data)
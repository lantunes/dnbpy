from __future__ import division, print_function, absolute_import

from enum import Enum
from params import mlp_params
import pandas as pd
import csv
import tensorflow as tf
import numpy as np
import random
from params import Type
import utils
from utils import *
from dnbpy import GameEngine

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data


SEED = 2016
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
            self._keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)
            self._init_weights_and_biases(self._params) #initialize the graph and required parameters
            self.logits = self.multilayer_perceptron_prediction(self._input) #Apply MLP map to the input and get output values
            #self.prediction = tf.nn.softmax(self.logits)
            self.prediction = self.softmax_with_valid_moves(self.logits,self._input)

            # Define loss and optimizer
            self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self._output))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self._params.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss_op)
            self.correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self._output, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
            # Evaluate model (write proper optimization)
            #self.correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self._output, 1))
            #self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
            self.init_op = tf.initialize_all_variables()


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

    def mlp_pg_train(self,states,actions,rewards):
        """

        :param states:
        :param actions:
        :param rewards:
        :return:
        """
        pass

    #def policy_loss(self):

    def train_mlp_policy(self,data):
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
            #reset data-indexes
            reset_data_indexes()

            index_list = list(set(shuffeled_data['index']))
            #Create train/test folds
            print("Folding the data into train and test sets")
            folds = n_fold_cross_validation(shuffeled_data, self._params.num_folds, index_list)
            fold_index = 0

            #for fold_index, fold in enumerate(folds):
                #print("Fold:"+str(fold_index)+"....")
                #create train and test data-frames
            train_df, test_df = generate_train_validation_form_fold(fold_index, folds)
            #extract train/test states and labels
            train_states = list(train_df['state'])
            train_labels = list(train_df['move'])
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
                    #average_loss = 0


def get_random_index(current_state):
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
    elif agent_type==agent.RANDOM:
        #Uniformly sample from valid moves
        next_move = get_random_index(current_state)
        pass

    return (next_move)



if __name__=="__main__":

    mlp_object = mlp_params(3, 3, [])
    mlp_policy = mlp_policy_network(mlp_object)

    # Initial state
    rewards = []
    reward_saver = []
    player_to_update = 0

    #Big loop over all the episodes
    for episode_index in range(0, 1):
        #tempreture = gen_learning_rate(episode_index, TEMPRETURE, .01, NUM_EPISODES)

        #Choose the player_index first
        if random.random() < .5:
            player_to_update = 0
        else:
            player_to_update = 1

        # Launch the game-engine
        engine = GameEngine((3, 3), (0, 1))
        #get the current state of the game
        current_state = engine._board_state
        #initialize backups
        backups = {}
        #game_condition is True for now
        game_condition = True
        while game_condition:
            #This is the current player_index
            player_index = engine.get_current_player()
            #Run one-step sample, use the MLP-network and get the current state and produce output probabilities; use tempreture-wise sampling
            if player_index==player_to_update:
                line_index = one_shot_board_play(current_state, mlp_policy, 1,agent.MLP)
            else:
                line_index = one_shot_board_play(current_state,mlp_policy,1,agent.RANDOM)
            #make a movement
            engine.select_edge(line_index, player_index)
            #check for the status of the game
            if engine.is_game_finished():
                game_condition = False
                break
            current_state = engine.get_board_state()
            #store visited states + actions for policy update
            if not backups.has_key(player_index): backups[player_index] = []
            backups[player_index].append([current_state,line_index])  # store all the visited states and actions as well

        # Get the final reward
        scores = []
        scores.append(engine.get_score(0))
        scores.append(engine.get_score(1))
        # Determine reward
        if scores[player_to_update] > scores[abs(1 - player_to_update)]:
            reward = 1
        else:
            reward = -1

        #Update_part (update parameters of the MLP network using a simple Policy-gradient rule)

        sys.exit(1)
        if episode_index % 500 == 0 and episode_index > 0:
            reward_saver.append(sum(rewards))
            print(episode_index, tempreture, sum(rewards))
            rewards = []
            # print(reward_saver)
        else:
            rewards.append(reward)



    #input = [0]*24
    #input[0] = 1
    #input[10] = 1
    #print(mlp_policy.mlp_play(np.array(input).reshape([1,24])))

    sys.exit(1)
    #Read training data and extract states and labels
    data_file = "data/dnb_moves_indexed_dedup.txt"
    data = pd.read_csv(data_file, sep=",", quoting=csv.QUOTE_ALL)
    mlp_policy.train_mlp_policy(data)

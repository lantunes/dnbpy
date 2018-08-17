from ai import *
from dnbpy import *
import tensorflow as tf
import numpy as np
from util.initializer_util import *


class PGPolicy3x3CNN(Policy):
    def __init__(self, board_size, n_channels = 1, batch_size=1, existing_params=None, dropout_keep_prob=1.0, activation=tf.nn.relu):
        self._board_size = board_size
        self._batch_size = batch_size
        self._n_channels = n_channels
        self._epsilon = 0.0
        self._temperature = 0.0
        self._dropout_keep_prob = dropout_keep_prob
        self._activation = activation

        edge_matrix = init_edge_matrix(board_size)
        self._n_input_rows = edge_matrix.shape[0]
        self._n_input_cols = edge_matrix.shape[1]
        self._n_hidden = 500
        self._n_output = len(init_board_state(board_size))

        # TF graph creation
        g = tf.Graph()
        with g.as_default():
            #self._input = tf.placeholder("float", [None, self._n_input_rows, self._n_input_cols,self._n_channels], name="input")
            self._input = tf.placeholder("float", [None, self._n_input_rows, self._n_input_cols*self._n_channels],
                                         name="input")
            self._action_taken = tf.placeholder("float", [None, self._n_output], name="action_taken")
            self._outcome = tf.placeholder(tf.float32, (None, 1), name="outcome")
            self._lr = tf.placeholder("float", shape=[], name="learning_rate")
            self._keep_prob = tf.placeholder(tf.float32)

            W_in_initializer = existing_param_initializer(existing_params, "W_in", tf.random_normal_initializer(0.0, 0.1))
            b_in_initializer = existing_param_initializer(existing_params, "b_in", tf.zeros_initializer())
            W_out_initializer = existing_param_initializer(existing_params, "W_out", tf.random_normal_initializer(0.0, 0.1))
            conv_kernel_initializer = existing_param_initializer(existing_params, "conv2d/kernel:0", tf.random_normal_initializer(0.0, 0.1))
            conv_bias_initializer = existing_param_initializer(existing_params, "conv2d/bias:0", tf.zeros_initializer())
            conv_kernel2_initializer = existing_param_initializer(existing_params, "conv2d_1/kernel:0", tf.random_normal_initializer(0.0, 0.1))
            conv_bias2_initializer = existing_param_initializer(existing_params, "conv2d_1/bias:0", tf.zeros_initializer())

            self._W_in = tf.get_variable(shape=[5 * 5 * 24, self._n_hidden], initializer=W_in_initializer, name="W_in")
            self._b_in = tf.get_variable(shape=[self._n_hidden], initializer=b_in_initializer, name="b_in")
            self._W_out = tf.get_variable(shape=[self._n_hidden, self._n_output], initializer=W_out_initializer, name="W_out")

            self._input_reshaped = tf.reshape(self._input, shape=[tf.shape(self._input)[0], self._n_input_rows, self._n_input_cols, self._n_channels])

            # Convolutional Layer 1
            # Computes 12 features using a 3x3 filter with ReLU activation.
            # Padding is added to preserve width and height.
            # Input Tensor Shape (for the 3x3 board): [None, 7, 7, 1] (batch size, width, height, channels)
            # Output Tensor Shape: [None, 7, 7, 12]
            self._conv = tf.layers.conv2d(
                inputs=self._input_reshaped,
                filters=12,
                kernel_size=[3, 3],
                strides=(1, 1),
                padding="same",
                kernel_initializer=conv_kernel_initializer,
                bias_initializer=conv_bias_initializer,
                activation=self._activation)

            # Convolutional Layer 2
            # Computes 24 features using a 3x3 filter with ReLU activation.
            # No padding is added.
            # Input Tensor Shape: [None, 7, 7, 12]
            # Output Tensor Shape: [None, 5, 5, 24]
            self._conv2 = tf.layers.conv2d(
                inputs=self._conv,
                filters=24,
                kernel_size=[3, 3],
                strides=(1, 1),
                kernel_initializer=conv_kernel2_initializer,
                bias_initializer=conv_bias2_initializer,
                activation=self._activation)

            self._conv_flat = tf.reshape(self._conv2, [tf.shape(self._input)[0], 5 * 5 * 24])

            dense_layer = tf.nn.tanh(tf.matmul(self._conv_flat, self._W_in) + self._b_in)

            drop_out = tf.nn.dropout(dense_layer, self._keep_prob)

            # TODO: tf.matmul(drop_out, self._W_out) should be divided by a temperature (which is 1.0 by default)
            self._action_probs = tf.nn.softmax(tf.matmul(drop_out, self._W_out))

            self._cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=tf.matmul(drop_out, self._W_out), labels=self._action_taken)
            self._loss = self._cross_entropy * self._outcome
            if self._batch_size > 1:
                self._loss = tf.reduce_mean(self._loss)
            self._train_op = tf.train.GradientDescentOptimizer(self._lr).minimize(self._loss)

            self._conv2d_kernel = [v for v in tf.global_variables() if v.name == 'conv2d/kernel:0'][0]
            self._conv2d_bias = [v for v in tf.global_variables() if v.name == 'conv2d/bias:0'][0]
            self._conv2d_kernel2 = [v for v in tf.global_variables() if v.name == 'conv2d_1/kernel:0'][0]
            self._conv2d_bias2 = [v for v in tf.global_variables() if v.name == 'conv2d_1/bias:0'][0]

        # TF session creation and initialization
        self._sess = tf.Session(graph=g, config=tf.ConfigProto(use_per_session_threads=True))
        with g.as_default():
            self._sess.run(tf.global_variables_initializer())

    def get_architecture(self):
        return "7x7-conv(3x3, %s, 12)-conv(3x3, %s, 24)-tanh(500)-softmax(1)" % \
               (self._activation.__name__, self._activation.__name__)

    def select_edge(self, board_state, tensor_state):
        #edge_matrix = convert_board_state_to_edge_matrix(self._board_size, board_state)
        #edge_matrix = tensor_state[0]
        state = np.concatenate(tensor_state,axis=1)
        action_probs = self._sess.run([self._action_probs], feed_dict={
            self._input: [state],
            self._keep_prob: 1.0
        })
        # convert to array
        action_probs = action_probs[0][0]

        zero_indices = []  # indices of legal actions
        for i in range(len(board_state)):
            if board_state[i] == 0:
                zero_indices.append(i)

        if self._boltzmann_action:
            # return the action sampled according to temperature and the given probs
            # re-normalize the probs to disallow illegal actions
            legal_raw_probs = []
            for z in zero_indices:
                legal_raw_probs.append(action_probs[z])
            legal_normalized_probs = self._softmax(legal_raw_probs)
            boltzmann = lambda x, T : np.exp(x/T)/np.sum(np.exp(x/T))
            selected_index = np.random.choice(zero_indices, p=boltzmann(np.array(legal_normalized_probs), self._temperature))
            return selected_index
        else:
            if random.random() < self._epsilon:
                # do epsilon greedy action
                return random.choice(zero_indices)
            else:
                # return the legal action with the highest prob
                highest_prob_index = random.choice(zero_indices)
                highest_prob = action_probs[highest_prob_index]
                for i, p in enumerate(action_probs):
                    if i in zero_indices and p > highest_prob:
                        highest_prob_index = i
                        highest_prob = p
                return highest_prob_index

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def get_action_probs_input_channel(self, board_state,tensor_state, normalize_with_softmax=False):
        #edge_matrix = convert_board_state_to_edge_matrix(self._board_size, board_state)
        edge_matrix = tensor_state[0]
        input_tensor = np.concatenate(tensor_state,axis=1)
        action_probs = self._sess.run([self._action_probs], feed_dict={
            self._input: [input_tensor],
            self._keep_prob: 1.0
        })
        # convert to ndarray
        action_probs = action_probs[0][0]

        zero_indices = []  # indices of legal actions
        for i in range(len(board_state)):
            if board_state[i] == 0:
                zero_indices.append(i)

        legal_raw_probs = []
        for z in zero_indices:
            legal_raw_probs.append(action_probs[z])
        if normalize_with_softmax:
            legal_normalized_probs = self._softmax(legal_raw_probs)
        else:
            legal_normalized_probs = self._normalize(legal_raw_probs)

        action_prob_map = {}
        for i in range(len(zero_indices)):
            action_state = [x for x in board_state]
            action_state[zero_indices[i]] = 1
            action_prob_map[as_string(action_state)] = legal_normalized_probs[i]
        return action_prob_map

    def get_action_probs(self, board_state, normalize_with_softmax=False):
        edge_matrix = convert_board_state_to_edge_matrix(self._board_size, board_state)
        action_probs = self._sess.run([self._action_probs], feed_dict={
            self._input: [edge_matrix],
            self._keep_prob: 1.0
        })
        # convert to ndarray
        action_probs = action_probs[0][0]

        zero_indices = []  # indices of legal actions
        for i in range(len(board_state)):
            if board_state[i] == 0:
                zero_indices.append(i)

        legal_raw_probs = []
        for z in zero_indices:
            legal_raw_probs.append(action_probs[z])
        if normalize_with_softmax:
            legal_normalized_probs = self._softmax(legal_raw_probs)
        else:
            legal_normalized_probs = self._normalize(legal_raw_probs)

        action_prob_map = {}
        for i in range(len(zero_indices)):
            action_state = [x for x in board_state]
            action_state[zero_indices[i]] = 1
            action_prob_map[as_string(action_state)] = legal_normalized_probs[i]
        return action_prob_map

    def _normalize(self, probs):
        prob_factor = 1 / sum(probs)
        return np.array([prob_factor * p for p in probs])

    def get_temperature(self):
        return self._temperature

    def set_temperature(self, temperature):
        self._temperature = temperature

    def is_boltzmann_action(self):
        return self._boltzmann_action

    def set_boltzmann_action(self, is_boltzmann_action):
        self._boltzmann_action = is_boltzmann_action

    def get_epsilon(self):
        return self._epsilon

    def set_epsilon(self, eps):
        self._epsilon = eps

    def get_learning_rate(self):
        return self._learning_rate

    def set_learning_rate(self, lr):
        self._learning_rate = lr

    def update_model(self, transitions):
        np.random.shuffle(transitions)
        batches = list(self._minibatches(transitions, batch_size=self._batch_size))
        for b in range(len(batches)):
            batch = batches[b]
            #states = [convert_board_state_to_edge_matrix(self._board_size, row[0]) for row in batch]
            states = [np.concatenate(row[0],axis=1) for row in batch]
            #states = np.concatenate(states,axis=0)
            actions = [row[1] for row in batch]
            outcomes = [[row[2]] for row in batch]
            self._sess.run([self._train_op], {
                self._input:        states,
                self._action_taken: actions,
                self._outcome:      outcomes,
                self._lr:           self._learning_rate,
                self._keep_prob:    self._dropout_keep_prob
            })

    def _minibatches(self, samples, batch_size):
        for i in range(0, len(samples), batch_size):
            yield samples[i:i + batch_size]

    def get_params(self):
        params_map = {}
        params = self._sess.run([self._W_in])
        params_map["W_in"] = params[0].tolist()
        params = self._sess.run([self._b_in])
        params_map["b_in"] = params[0].tolist()
        params = self._sess.run([self._W_out])
        params_map["W_out"] = params[0].tolist()
        params = self._sess.run([self._conv2d_kernel])
        params_map["conv2d/kernel:0"] = params[0].tolist()
        params = self._sess.run([self._conv2d_bias])
        params_map["conv2d/bias:0"] = params[0].tolist()
        params = self._sess.run([self._conv2d_kernel2])
        params_map["conv2d_1/kernel:0"] = params[0].tolist()
        params = self._sess.run([self._conv2d_bias2])
        params_map["conv2d_1/bias:0"] = params[0].tolist()
        return params_map

    def copy(self):
        policy_copy = type(self)(self._board_size,self._n_channels, self._batch_size, self.get_params())
        policy_copy.set_temperature(self.get_temperature())
        policy_copy.set_epsilon(self.get_epsilon())
        policy_copy.set_boltzmann_action(self.is_boltzmann_action())
        return policy_copy

    def print_params(self, f):
        params = self._sess.run([self._W_in])
        f.write("W_in: %s\n" % params[0].tolist())
        params = self._sess.run([self._b_in])
        f.write("b_in: %s\n" % params[0].tolist())
        params = self._sess.run([self._W_out])
        f.write("W_out: %s\n" % params[0].tolist())
        params = self._sess.run([self._conv2d_kernel])
        f.write("conv2d/kernel:0: %s\n" % params[0].tolist())
        params = self._sess.run([self._conv2d_bias])
        f.write("conv2d/bias:0: %s\n" % params[0].tolist())
        params = self._sess.run([self._conv2d_kernel2])
        f.write("conv2d_1/kernel:0: %s\n" % params[0].tolist())
        params = self._sess.run([self._conv2d_bias2])
        f.write("conv2d_1/bias:0: %s\n" % params[0].tolist())
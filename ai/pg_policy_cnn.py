from ai import *
from dnbpy import *
import tensorflow as tf
import numpy as np


class PGPolicyCNN(Policy):
    def __init__(self, board_size, batch_size=1):
        self._sess = tf.Session()
        self._board_size = board_size
        self._batch_size = batch_size

        edge_matrix = init_edge_matrix(board_size)
        self._n_input_rows = edge_matrix.shape[0]
        self._n_input_cols = edge_matrix.shape[1]
        self._n_hidden = 300
        self._n_output = len(init_board_state(board_size))

        self._input = tf.placeholder("float", [None, self._n_input_rows, self._n_input_cols], name="input")
        self._action_taken = tf.placeholder("float", [None, self._n_output], name="action_taken")
        self._outcome = tf.placeholder(tf.float32, (None, 1), name="outcome")
        self._lr = tf.placeholder("float", shape=[], name="learning_rate")

        self._W_in = tf.Variable(tf.random_normal([5 * 5 * 12, self._n_hidden], 0.0, 0.1), name="W_in")
        self._b_in = tf.Variable(tf.zeros([self._n_hidden]), name="b_in")
        self._W_out = tf.Variable(tf.random_normal([self._n_hidden, self._n_output], 0.0, 0.1), name="W_out")

        self._input_reshaped = tf.reshape(self._input, shape=[tf.shape(self._input)[0], self._n_input_rows, self._n_input_cols, 1])

        # Convolutional Layer
        # Computes 12 features using a 3x3 filter with ReLU activation.
        # Padding is added to preserve width and height.
        # Input Tensor Shape (for the 2x2 board): [None, 5, 5, 1] (batch size, width, height, channels)
        # Output Tensor Shape: [None, 5, 5, 12]
        self._conv = tf.layers.conv2d(
            inputs=self._input_reshaped,
            filters=12,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            kernel_initializer=tf.random_normal_initializer(0.0, 0.1),
            activation=tf.nn.relu)

        self._conv_flat = tf.reshape(self._conv, [tf.shape(self._input)[0], 5 * 5 * 12])

        dense_layer = tf.nn.tanh(tf.matmul(self._conv_flat, self._W_in) + self._b_in)

        self._action_probs = tf.nn.softmax(tf.matmul(dense_layer, self._W_out))

        self._cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=tf.matmul(dense_layer, self._W_out), labels=self._action_taken)

        self._loss = self._cross_entropy * self._outcome

        self._train_op = tf.train.GradientDescentOptimizer(self._lr).minimize(self._loss)

        self._conv2d_kernel = [v for v in tf.global_variables() if v.name == 'conv2d/kernel:0'][0]
        self._conv2d_bias = [v for v in tf.global_variables() if v.name == 'conv2d/bias:0'][0]

        self._sess.run(tf.global_variables_initializer())

    def get_architecture(self):
        return "5x5-conv(3x3, relu, 12)-tanh(300)-sigmoid(1)"

    def select_edge(self, board_state):
        edge_matrix = convert_board_state_to_edge_matrix(self._board_size, board_state)
        action_probs = self._sess.run([self._action_probs], feed_dict={self._input: [edge_matrix]})
        # convert to 12D ndarray
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
            # return the legal action with the highest prob
            highest_prob_index = 0
            highest_prob = 0.0
            for i, p in enumerate(action_probs):
                if i in zero_indices and p > highest_prob:
                    highest_prob_index = i
                    highest_prob = p
            return highest_prob_index

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def get_temperature(self):
        return self._temperature

    def set_temperature(self, temperature):
        self._temperature = temperature

    def is_boltzmann_action(self):
        return self._boltzmann_action

    def set_boltzmann_action(self, is_boltzmann_action):
        self._boltzmann_action = is_boltzmann_action

    def get_learning_rate(self):
        return self._learning_rate

    def set_learning_rate(self, lr):
        self._learning_rate = lr

    def update_model(self, transitions):
        np.random.shuffle(transitions)
        batches = list(self._minibatches(transitions, batch_size=self._batch_size))
        for b in range(len(batches)):
            batch = batches[b]
            states = [convert_board_state_to_edge_matrix(self._board_size, row[0]) for row in batch]
            actions = [row[1] for row in batch]
            outcomes = [[row[2]] for row in batch]
            self._sess.run([self._train_op], {
                self._input:        states,
                self._action_taken: actions,
                self._outcome:      outcomes,
                self._lr:           self._learning_rate
            })

    def _minibatches(self, samples, batch_size):
        for i in range(0, len(samples), batch_size):
            yield samples[i:i + batch_size]

    def print_params(self, f):
        params = self._sess.run([self._W_in])
        f.write("W_in: %s\n" % params[0].tolist())
        params = self._sess.run([self._b_in])
        f.write("b_in: %s\n" % params[0].tolist())
        params = self._sess.run([self._W_out])
        f.write("W_out: %s\n" % params[0].tolist())
        params = self._sess.run([self._conv2d_kernel])
        f.write("conv2d_kernel: %s\n" % params[0].tolist())
        params = self._sess.run([self._conv2d_bias])
        f.write("conv2d_bias: %s\n" % params[0].tolist())

from dnbpy.ai import *
from dnbpy import *
import tensorflow as tf
import numpy as np
from dnbpy.util.initializer_util import *


class SLValue3x3CNN(Policy):
    def __init__(self, board_size, batch_size=1, existing_params=None, dropout_keep_prob=1.0, activation=tf.nn.relu):
        self._board_size = board_size
        self._batch_size = batch_size
        self._epsilon = 0.0
        self._temperature = 0.0
        self._dropout_keep_prob = dropout_keep_prob
        self._activation = activation

        self._edge_length = 1
        self._include_dots = True

        edge_matrix = init_edge_matrix(board_size, edge_length=self._edge_length, include_dots=self._include_dots)
        self._n_input_rows = edge_matrix.shape[0]
        self._n_input_cols = edge_matrix.shape[1]
        self._n_hidden = 500
        self._n_output = 1

        # TF graph creation
        g = tf.Graph()
        with g.as_default():
            self._input = tf.placeholder("float", [None, self._n_input_rows, self._n_input_cols], name="input")
            self._outcome = tf.placeholder(tf.float32, (None, 1), name="outcome")
            self._lr = tf.placeholder("float", shape=[], name="learning_rate")
            self._keep_prob = tf.placeholder(tf.float32, name="keep_prob")

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

            self._input_reshaped = tf.reshape(self._input, shape=[tf.shape(self._input)[0], self._n_input_rows, self._n_input_cols, 1])

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

            self._prediction = tf.nn.sigmoid(tf.matmul(drop_out, self._W_out))

            # regression losses
            # self._loss = tf.square(self._outcome - self._prediction, name="loss")
            # self._loss = -self._outcome*tf.log(self._prediction) - (1 - self._outcome)*tf.log(1 - self._prediction)  # KL Loss: −z log[V(s)] − (1 − z) log[1 − V(s)]
            # classification loss
            self._loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.matmul(drop_out, self._W_out),
                                                                 labels=self._outcome)

            if self._batch_size > 1:
                self._loss = tf.reduce_mean(self._loss)
            self._train_op = tf.train.AdamOptimizer(self._lr).minimize(self._loss)

            self._conv2d_kernel = [v for v in tf.global_variables() if v.name == 'conv2d/kernel:0'][0]
            self._conv2d_bias = [v for v in tf.global_variables() if v.name == 'conv2d/bias:0'][0]
            self._conv2d_kernel2 = [v for v in tf.global_variables() if v.name == 'conv2d_1/kernel:0'][0]
            self._conv2d_bias2 = [v for v in tf.global_variables() if v.name == 'conv2d_1/bias:0'][0]

        # TF session creation and initialization
        self._sess = tf.Session(graph=g, config=tf.ConfigProto(use_per_session_threads=True))
        with g.as_default():
            self._sess.run(tf.global_variables_initializer())

    def get_architecture(self):
        return "7x7-conv(3x3, %s, 12)-conv(3x3, %s, 24)-tanh(500)-sigmoid(1)" % \
               (self._activation.__name__, self._activation.__name__)

    def select_edge(self, board_state):
        zero_indices = []
        for i in range(len(board_state)):
            if board_state[i] == 0:
                zero_indices.append(i)
        if random.random() < self._epsilon:
            chosen_index = random.choice(zero_indices)
            return chosen_index
        else:
            best_value = 0.0
            best_state_index = None
            for zero_index in zero_indices:
                new_state = [x for x in board_state]
                new_state[zero_index] = 1
                new_state = convert_board_state_to_edge_matrix(self._board_size, new_state,
                                                               edge_length=self._edge_length,
                                                               include_dots=self._include_dots)
                new_state_value = self._sess.run([self._prediction], feed_dict={
                    self._input: [new_state],
                    self._keep_prob: 1.0
                })
                if best_state_index is None or new_state_value > best_value:
                    best_value = new_state_value
                    best_state_index = zero_index
            return best_state_index

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
            states = [convert_board_state_to_edge_matrix(self._board_size, row[0], edge_length=self._edge_length,
                                                         include_dots=self._include_dots) for row in batch]
            outcomes = [[row[1]] for row in batch]
            self._sess.run([self._train_op], {
                self._input:        states,
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
        policy_copy = type(self)(self._board_size, self._batch_size, self.get_params(), self._dropout_keep_prob, self._activation)
        policy_copy.set_epsilon(self.get_epsilon())
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

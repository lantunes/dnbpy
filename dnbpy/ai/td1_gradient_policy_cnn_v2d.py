from dnbpy.ai import *
from dnbpy import *
import tensorflow as tf
import numpy as np


class TDOneGradientPolicyCNNV2d(Policy):
    """
    Adds padding to the initial convolutional layer, and a second convolutional layer.
    """
    def __init__(self, board_size):
        self._sess = tf.Session()
        self._board_size = board_size

        edge_matrix = init_edge_matrix(board_size)
        self._n_input_rows = edge_matrix.shape[0]
        self._n_input_cols = edge_matrix.shape[1]
        self._n_hidden = 300
        self._n_output = 1

        self._input = tf.placeholder("float", [self._n_input_rows, self._n_input_cols], name="input")
        self._target = tf.placeholder("float", [1, self._n_output], name="target")
        self._error = tf.placeholder("float", shape=[], name="error")
        self._lr = tf.placeholder("float", shape=[], name="learning_rate")
        self._sum_grad_W_in = tf.placeholder("float", shape=[3 * 3 * 24, self._n_hidden], name="sum_grad_W_in")
        self._sum_grad_b_in = tf.placeholder("float", shape=[self._n_hidden], name="sum_grad_b_in")
        self._sum_grad_W_out = tf.placeholder("float", shape=[self._n_hidden, 1], name="sum_grad_W_out")
        self._sum_conv2d_kernel = tf.placeholder("float", shape=[3, 3, 1, 12], name="sum_conv2d_kernel")
        self._sum_conv2d_bias = tf.placeholder("float", shape=[12], name="sum_conv2d_bias")
        self._sum_conv2d_kernel2 = tf.placeholder("float", shape=[3, 3, 12, 24], name="sum_conv2d_kernel2")
        self._sum_conv2d_bias2 = tf.placeholder("float", shape=[24], name="sum_conv2d_bias2")

        self._W_in = tf.Variable(tf.random_normal([3 * 3 * 24, self._n_hidden], 0.0, 0.1), name="W_in")
        self._b_in = tf.Variable(tf.zeros([self._n_hidden]), name="b_in")
        self._W_out = tf.Variable(tf.random_normal([self._n_hidden, self._n_output], 0.0, 0.1), name="W_out")

        self._input_reshaped = tf.reshape(self._input, shape=[1, self._n_input_rows, self._n_input_cols, 1])

        # Convolutional Layer 1
        # Computes 12 features using a 3x3 filter with ReLU activation.
        # Padding is added to preserve width and height.
        # Input Tensor Shape (for the 2x2 board): [1, 5, 5, 1] (batch size, width, height, channels)
        # Output Tensor Shape: [1, 5, 5, 12]
        self._conv = tf.layers.conv2d(
            inputs=self._input_reshaped,
            filters=12,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            kernel_initializer=tf.random_normal_initializer(0.0, 0.1),
            activation=tf.nn.relu)

        # Convolutional Layer 2
        # Computes 24 features using a 3x3 filter with ReLU activation.
        # No padding is added.
        # Input Tensor Shape: [1, 5, 5, 12]
        # Output Tensor Shape: [1, 3, 3, 24]
        self._conv2 = tf.layers.conv2d(
            inputs=self._conv,
            filters=24,
            kernel_size=[3, 3],
            strides=(1, 1),
            kernel_initializer=tf.random_normal_initializer(0.0, 0.1),
            activation=tf.nn.relu)

        self._conv_flat = tf.reshape(self._conv2, [1, 3 * 3 * 24])

        dense_layer = tf.nn.tanh(tf.matmul(self._conv_flat, self._W_in) + self._b_in)

        self._prediction = tf.nn.sigmoid(tf.matmul(dense_layer, self._W_out))

        self._conv2d_kernel = [v for v in tf.global_variables() if v.name == 'conv2d/kernel:0'][0]
        self._conv2d_bias = [v for v in tf.global_variables() if v.name == 'conv2d/bias:0'][0]
        self._conv2d_kernel2 = [v for v in tf.global_variables() if v.name == 'conv2d_1/kernel:0'][0]
        self._conv2d_bias2 = [v for v in tf.global_variables() if v.name == 'conv2d_1/bias:0'][0]

        self._gradients = tf.gradients(self._prediction, [self._W_in, self._b_in, self._W_out,
                                                          self._conv2d_kernel, self._conv2d_bias,
                                                          self._conv2d_kernel2, self._conv2d_bias2])

        self._update_W_in = self._W_in.assign(self._W_in + self._lr * self._error * self._sum_grad_W_in)
        self._update_b_in = self._b_in.assign(self._b_in + self._lr * self._error * self._sum_grad_b_in)
        self._update_W_out = self._W_out.assign(self._W_out + self._lr * self._error * self._sum_grad_W_out)
        self._update_conv2d_kernel = self._conv2d_kernel.assign(self._conv2d_kernel + self._lr * self._error * self._sum_conv2d_kernel)
        self._update_conv2d_bias = self._conv2d_bias.assign(self._conv2d_bias + self._lr * self._error * self._sum_conv2d_bias)
        self._update_conv2d_kernel2 = self._conv2d_kernel2.assign(self._conv2d_kernel2 + self._lr * self._error * self._sum_conv2d_kernel2)
        self._update_conv2d_bias2 = self._conv2d_bias2.assign(self._conv2d_bias2 + self._lr * self._error * self._sum_conv2d_bias2)

        self._sess.run(tf.global_variables_initializer())

        self.reset_history_buffer()

    def get_architecture(self):
        return "5x5-conv(3x3, relu, 12)-conv(3x3, relu, 24)-tanh(300)-sigmoid(1)"

    def reset_history_buffer(self):
        self._prediction_buffer = []
        self._prediction_gradient_buffer = []

    def get_last_prediction(self):
        if len(self._prediction_buffer) > 0:
            return self._prediction_buffer[-1]

    def get_last_prediction_gradient(self):
        if len(self._prediction_gradient_buffer) > 0:
            return self._prediction_gradient_buffer[-1]

    def select_edge(self, board_state):
        zero_indices = []
        for i in range(len(board_state)):
            if board_state[i] == 0:
                zero_indices.append(i)
        if random.random() < self._epsilon:
            random_index = random.choice(zero_indices)
            # store history
            new_state = [x for x in board_state]
            new_state[random_index] = 1
            new_state = convert_board_state_to_edge_matrix(self._board_size, new_state)
            new_state_value, gradients = self._sess.run([self._prediction, self._gradients],
                                                        feed_dict={self._input: new_state})
            self._prediction_buffer.append(new_state_value[0][0])
            self._prediction_gradient_buffer.append(gradients)
            return random_index
        else:
            best_value = 0.0
            best_value_gradient = None
            best_state_index = zero_indices[0]
            for zero_index in zero_indices:
                new_state = [x for x in board_state]
                new_state[zero_index] = 1
                new_state = convert_board_state_to_edge_matrix(self._board_size, new_state)
                new_state_value, gradients = self._sess.run([self._prediction, self._gradients],
                                                            feed_dict={self._input: new_state})
                if new_state_value >= best_value:
                    best_value = new_state_value
                    best_value_gradient = gradients
                    best_state_index = zero_index
            # store history
            self._prediction_buffer.append(best_value[0][0])
            self._prediction_gradient_buffer.append(best_value_gradient)
            return best_state_index

    def get_epsilon(self):
        return self._epsilon

    def set_epsilon(self, eps):
        self._epsilon = eps

    def get_learning_rate(self):
        return self._learning_rate

    def set_learning_rate(self, lr):
        self._learning_rate = lr

    def update(self, prediction_history, prediction_gradient_history):
        if len(prediction_history) > 1:
            error = prediction_history[-1] - prediction_history[-2]
            sum_grad_W_in = np.sum(prediction_gradient_history[:-1], axis=0)[0]
            sum_grad_b_in = np.sum(prediction_gradient_history[:-1], axis=0)[1]
            sum_grad_W_out = np.sum(prediction_gradient_history[:-1], axis=0)[2]
            sum_conv2d_kernel = np.sum(prediction_gradient_history[:-1], axis=0)[3]
            sum_conv2d_bias = np.sum(prediction_gradient_history[:-1], axis=0)[4]
            sum_conv2d_kernel2 = np.sum(prediction_gradient_history[:-1], axis=0)[5]
            sum_conv2d_bias2 = np.sum(prediction_gradient_history[:-1], axis=0)[6]
            self._update_params(error, sum_grad_W_in, sum_grad_b_in, sum_grad_W_out, sum_conv2d_kernel, sum_conv2d_bias,
                                sum_conv2d_kernel2, sum_conv2d_bias2)

    def update_terminal(self, prediction_history, prediction_gradient_history, target):
        error = target - prediction_history[-1]
        sum_grad_W_in = np.sum(prediction_gradient_history, axis=0)[0]
        sum_grad_b_in = np.sum(prediction_gradient_history, axis=0)[1]
        sum_grad_W_out = np.sum(prediction_gradient_history, axis=0)[2]
        sum_conv2d_kernel = np.sum(prediction_gradient_history, axis=0)[3]
        sum_conv2d_bias = np.sum(prediction_gradient_history, axis=0)[4]
        sum_conv2d_kernel2 = np.sum(prediction_gradient_history, axis=0)[5]
        sum_conv2d_bias2 = np.sum(prediction_gradient_history, axis=0)[6]
        self._update_params(error, sum_grad_W_in, sum_grad_b_in, sum_grad_W_out, sum_conv2d_kernel, sum_conv2d_bias,
                            sum_conv2d_kernel2, sum_conv2d_bias2)

    def update_offline(self, prediction_history, prediction_gradient_history, target):
        if len(prediction_history) > 0:
            for i in range(1, len(prediction_history) + 1):
                prev = prediction_history[i - 1]
                last = prediction_history[i] if i < len(prediction_history) else target
                error = last - prev
                sum_grad_W_in = np.sum(prediction_gradient_history[:i], axis=0)[0]
                sum_grad_b_in = np.sum(prediction_gradient_history[:i], axis=0)[1]
                sum_grad_W_out = np.sum(prediction_gradient_history[:i], axis=0)[2]
                sum_conv2d_kernel = np.sum(prediction_gradient_history[:i], axis=0)[3]
                sum_conv2d_bias = np.sum(prediction_gradient_history[:i], axis=0)[4]
                sum_conv2d_kernel2 = np.sum(prediction_gradient_history[:i], axis=0)[5]
                sum_conv2d_bias2 = np.sum(prediction_gradient_history[:i], axis=0)[6]
                self._update_params(error, sum_grad_W_in, sum_grad_b_in, sum_grad_W_out, sum_conv2d_kernel,
                                    sum_conv2d_bias, sum_conv2d_kernel2, sum_conv2d_bias2)

    def _update_params(self, error, sum_grad_W_in, sum_grad_b_in, sum_grad_W_out, sum_conv2d_kernel, sum_conv2d_bias,
                       sum_conv2d_kernel2, sum_conv2d_bias2):
        self._sess.run([self._update_W_in, self._update_b_in, self._update_W_out,
                        self._update_conv2d_kernel, self._update_conv2d_bias],
                       feed_dict={self._lr: self._learning_rate, self._error: error,
                                  self._sum_grad_W_in: sum_grad_W_in, self._sum_grad_b_in: sum_grad_b_in,
                                  self._sum_grad_W_out: sum_grad_W_out, self._sum_conv2d_kernel: sum_conv2d_kernel,
                                  self._sum_conv2d_bias: sum_conv2d_bias, self._sum_conv2d_kernel2: sum_conv2d_kernel2,
                                  self._sum_conv2d_bias2: sum_conv2d_bias2})

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
        params = self._sess.run([self._conv2d_kernel2])
        f.write("conv2d_kernel2: %s\n" % params[0].tolist())
        params = self._sess.run([self._conv2d_bias2])
        f.write("conv2d_bias2: %s\n" % params[0].tolist())

    def print_gradients(self):
        print(self._prediction_gradient_buffer)

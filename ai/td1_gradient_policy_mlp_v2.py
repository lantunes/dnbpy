from ai import *
from dnbpy import *
import tensorflow as tf
import numpy as np


class TDOneGradientPolicyMLPV2(Policy):
    def __init__(self, board_size):
        self._sess = tf.Session()
        self._board_size = board_size

        self._n_input = len(init_board_state(board_size))
        self._n_hidden1 = 300
        self._n_hidden2 = 300
        self._n_hidden3 = 300
        self._n_output = 1

        self._input = tf.placeholder("float", [1, self._n_input], name="input")
        self._target = tf.placeholder("float", [1, self._n_output], name="target")
        self._error = tf.placeholder("float", shape=[], name="error")
        self._lr = tf.placeholder("float", shape=[], name="learning_rate")
        self._sum_grad_W1 = tf.placeholder("float", shape=[self._n_input, self._n_hidden1], name="sum_grad_W1")
        self._sum_grad_W2 = tf.placeholder("float", shape=[self._n_hidden1, self._n_hidden2], name="sum_grad_W2")
        self._sum_grad_W3 = tf.placeholder("float", shape=[self._n_hidden2, self._n_hidden3], name="sum_grad_W3")
        self._sum_grad_W_out = tf.placeholder("float", shape=[self._n_hidden3, self._n_output], name="sum_grad_W_out")

        self._W1 = tf.Variable(tf.random_uniform([self._n_input, self._n_hidden1], -1, 1), name="W1")
        self._W2 = tf.Variable(tf.random_uniform([self._n_hidden1, self._n_hidden2], -1, 1), name="W2")
        self._W3 = tf.Variable(tf.random_uniform([self._n_hidden2, self._n_hidden3], -1, 1), name="W3")
        self._W_out = tf.Variable(tf.random_normal([self._n_hidden3, self._n_output]), name="W_out")

        layer1 = tf.nn.tanh(tf.matmul(self._input, self._W1))
        layer2 = tf.nn.tanh(tf.matmul(layer1, self._W2))
        layer3 = tf.nn.tanh(tf.matmul(layer2, self._W3))

        self._prediction = tf.nn.sigmoid(tf.matmul(layer3, self._W_out))

        self._gradients = tf.gradients(self._prediction, [self._W1, self._W2, self._W3, self._W_out])

        self._update_W1 = self._W1.assign(self._W1 + self._lr * self._error * self._sum_grad_W1)
        self._update_W2 = self._W2.assign(self._W2 + self._lr * self._error * self._sum_grad_W2)
        self._update_W3 = self._W3.assign(self._W3 + self._lr * self._error * self._sum_grad_W3)
        self._update_W_out = self._W_out.assign(self._W_out + self._lr * self._error * self._sum_grad_W_out)

        self._sess.run(tf.global_variables_initializer())

        self.reset_history_buffer()

    def get_architecture(self):
        return "12-tanh(300)-tanh(300)-tanh(300)-sigmoid(1)"

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
            new_state = np.reshape(new_state, (1, len(new_state)))
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
                new_state = np.reshape(new_state, (1, len(new_state)))
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
            sum_grad_W1 = np.sum(prediction_gradient_history[:-1], axis=0)[0]
            sum_grad_W2 = np.sum(prediction_gradient_history[:-1], axis=0)[1]
            sum_grad_W3 = np.sum(prediction_gradient_history[:-1], axis=0)[2]
            sum_grad_W_out = np.sum(prediction_gradient_history[:-1], axis=0)[3]
            self._update_params(error, sum_grad_W1, sum_grad_W2, sum_grad_W3, sum_grad_W_out)

    def update_terminal(self, prediction_history, prediction_gradient_history, target):
        error = target - prediction_history[-1]
        sum_grad_W1 = np.sum(prediction_gradient_history, axis=0)[0]
        sum_grad_W2 = np.sum(prediction_gradient_history, axis=0)[1]
        sum_grad_W3 = np.sum(prediction_gradient_history, axis=0)[2]
        sum_grad_W_out = np.sum(prediction_gradient_history, axis=0)[3]
        self._update_params(error, sum_grad_W1, sum_grad_W2, sum_grad_W3, sum_grad_W_out)

    def update_offline(self, prediction_history, prediction_gradient_history, target):
        if len(prediction_history) > 0:
            for i in range(1, len(prediction_history) + 1):
                prev = prediction_history[i - 1]
                last = prediction_history[i] if i < len(prediction_history) else target
                error = last - prev
                sum_grad_W1 = np.sum(prediction_gradient_history[:i], axis=0)[0]
                sum_grad_W2 = np.sum(prediction_gradient_history[:i], axis=0)[1]
                sum_grad_W3 = np.sum(prediction_gradient_history[:i], axis=0)[2]
                sum_grad_W_out = np.sum(prediction_gradient_history[:i], axis=0)[3]
                self._update_params(error, sum_grad_W1, sum_grad_W2, sum_grad_W3, sum_grad_W_out)

    def _update_params(self, error, sum_grad_W1, sum_grad_W2, sum_grad_W3, sum_grad_W_out):
        self._sess.run([self._update_W1, self._update_W2, self._update_W3, self._update_W_out],
                       feed_dict={self._lr: self._learning_rate, self._error: error,
                                  self._sum_grad_W1: sum_grad_W1,
                                  self._sum_grad_W2: sum_grad_W2,
                                  self._sum_grad_W3: sum_grad_W3,
                                  self._sum_grad_W_out: sum_grad_W_out})

    def print_params(self):
        params = self._sess.run([self._W1])
        print("W1: %s" % params[0].tolist())
        params = self._sess.run([self._W2])
        print("W2: %s" % params[0].tolist())
        params = self._sess.run([self._W3])
        print("W3: %s" % params[0].tolist())
        params = self._sess.run([self._W_out])
        print("W_out: %s" % params[0].tolist())

    def print_gradients(self):
        print(self._prediction_gradient_buffer)

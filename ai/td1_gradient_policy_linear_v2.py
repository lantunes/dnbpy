from ai import *
from dnbpy import *
import tensorflow as tf
import numpy as np


class TDOneGradientLinearPolicyV2(Policy):
    def __init__(self, board_size):
        self._sess = tf.Session()
        self._board_size = board_size

        self._n_input = len(init_board_state(board_size))
        self._n_output = 1

        self._input = tf.placeholder("float", [1, self._n_input], name="input")
        self._target = tf.placeholder("float", [1, self._n_output], name="target")
        self._error = tf.placeholder("float", shape=[], name="error")
        self._lr = tf.placeholder("float", shape=[], name="learning_rate")
        self._sum_grad_W = tf.placeholder("float", shape=[self._n_input, self._n_output], name="sum_grad_W_in")

        # self._W = tf.Variable(tf.random_normal([self._n_input, self._n_output]), name="W_out")
        self._W = tf.Variable(tf.random_uniform([self._n_input, self._n_output], minval=-1, maxval=1), name="W_out")

        self._prediction = tf.nn.sigmoid(tf.matmul(self._input, self._W))
        # self._prediction = tf.matmul(self._input, self._W)

        self._gradients = tf.gradients(self._prediction, [self._W])

        self._update_W = self._W.assign(self._W + self._lr * self._error * self._sum_grad_W)

        self._sess.run(tf.global_variables_initializer())

        self.reset_history_buffer()

    def get_architecture(self):
        return "12-sigmoid(1)"

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
            best_value_gradient = [np.zeros(shape=(self._n_input, 1))]
            best_state_index = zero_indices[0]
            for zero_index in zero_indices:
                new_state = [x for x in board_state]
                new_state[zero_index] = 1
                new_state = np.reshape(new_state, (1, len(new_state)))
                new_state_value, gradients = self._sess.run([self._prediction, self._gradients],
                                                            feed_dict={self._input: new_state})
                new_state_value = new_state_value[0][0]
                if new_state_value >= best_value:
                    best_value = new_state_value
                    best_value_gradient = gradients
                    best_state_index = zero_index
            # store history
            self._prediction_buffer.append(best_value)
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
            sum_grad_W = np.sum(prediction_gradient_history[:-1], axis=0)[0]
            self._sess.run([self._update_W],
                           feed_dict={self._lr: self._learning_rate, self._error: error, self._sum_grad_W: sum_grad_W})

    def update_terminal(self, prediction_history, prediction_gradient_history, target):
        error = target - prediction_history[-1]
        sum_grad_W = np.sum(prediction_gradient_history, axis=0)[0]
        self._sess.run([self._update_W],
                       feed_dict={self._lr: self._learning_rate, self._error: error, self._sum_grad_W: sum_grad_W})

    def update_offline(self, prediction_history, prediction_gradient_history, target):
        if len(prediction_history) > 0:
            for i in range(1, len(prediction_history) + 1):
                prev = prediction_history[i - 1]
                last = prediction_history[i] if i < len(prediction_history) else target
                error = last - prev
                sum_grad_W = np.sum(prediction_gradient_history[:i], axis=0)[0]
                self._sess.run([self._update_W],
                               feed_dict={self._lr: self._learning_rate, self._error: error, self._sum_grad_W: sum_grad_W})

    def print_params(self):
        params = self._sess.run([self._W])
        print(np.reshape(params[0], [1, self._n_input]).tolist())

    def print_gradients(self):
        print(self._prediction_gradient_buffer)

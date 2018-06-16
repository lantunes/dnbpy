from ai import *
from dnbpy import *
import tensorflow as tf
import numpy as np


class TDOneGradientPolicy(Policy):
    def __init__(self, board_size):
        self._sess = tf.Session()
        self._board_size = board_size

        self._n_input = len(init_board_state(board_size))
        self._n_hidden = 300
        self._n_output = 1

        self._input = tf.placeholder("float", [1, self._n_input], name="input")
        self._target = tf.placeholder("float", [1, self._n_output], name="target")
        self._error = tf.placeholder("float", shape=[], name="error")
        self._lr = tf.placeholder("float", shape=[], name="learning_rate")
        self._sum_grad_W_in = tf.placeholder("float", shape=[self._n_input, self._n_hidden], name="sum_grad_W_in")
        self._sum_grad_W_out = tf.placeholder("float", shape=[self._n_hidden, self._n_output], name="sum_grad_W_out")

        self._W_in = tf.Variable(tf.random_uniform([self._n_input, self._n_hidden], -1, 1), name="W_in")
        self._W_out = tf.Variable(tf.random_normal([self._n_hidden, self._n_output]), name="W_out")

        input_layer = tf.nn.tanh(tf.matmul(self._input, self._W_in))

        self._prediction = tf.nn.sigmoid(tf.matmul(input_layer, self._W_out))

        self._gradients = tf.gradients(self._prediction, [self._W_in, self._W_out])

        self._update_W_in = self._W_in.assign(self._W_in + self._lr * self._error * self._sum_grad_W_in)
        self._update_W_out = self._W_out.assign(self._W_out + self._lr * self._error * self._sum_grad_W_out)

        self._sess.run(tf.global_variables_initializer())

        self.reset_history()

    def get_architecture(self):
        return "12-tanh(300)-sigmoid(1)"

    def reset_history(self):
        self._prediction_history = []
        self._prediction_gradient_history = []

    def select_edge(self, board_state):
        if self._softmax_action:
            return self._select_edge_softmax(board_state)
        else:
            return self._select_edge_epsilon_greedily(board_state)

    def _select_edge_softmax(self, board_state):
        softmax = lambda x, T : np.exp(x/T)/np.sum(np.exp(x/T))
        zero_indices = []
        for i in range(len(board_state)):
            if board_state[i] == 0:
                zero_indices.append(i)
        new_state_values = []
        new_state_gradients = []
        for zero_index in zero_indices:
            new_state = [x for x in board_state]
            new_state[zero_index] = 1
            new_state = np.reshape(new_state, (1, len(new_state)))
            new_state_value, gradients = self._sess.run([self._prediction, self._gradients],
                                                        feed_dict={self._input: new_state})
            new_state_values.append(new_state_value[0][0])
            new_state_gradients.append(gradients)
        selected_index = np.argmax(np.random.multinomial(1, softmax(np.array(new_state_values), self._temperature)))
        if self._store_history:
            self._prediction_history.append(new_state_values[selected_index])
            self._prediction_gradient_history.append(new_state_gradients[selected_index])
        return zero_indices[selected_index]

    def _select_edge_epsilon_greedily(self, board_state):
        zero_indices = []
        for i in range(len(board_state)):
            if board_state[i] == 0:
                zero_indices.append(i)
        if random.random() < self._epsilon:
            chosen_index = random.choice(zero_indices)
            if self._store_history:
                new_state = [x for x in board_state]
                new_state[chosen_index] = 1
                new_state = np.reshape(new_state, (1, len(new_state)))
                new_state_value, gradients = self._sess.run([self._prediction, self._gradients],
                                                           feed_dict={self._input: new_state})
                self._prediction_history.append(new_state_value[0][0])
                self._prediction_gradient_history.append(gradients)
            return chosen_index
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
            if self._store_history:
                self._prediction_history.append(best_value[0][0])
                self._prediction_gradient_history.append(best_value_gradient)
            return best_state_index

    def set_should_store_history(self, store_history):
        self._store_history = store_history

    def get_epsilon(self):
        return self._epsilon

    def set_epsilon(self, eps):
        self._epsilon = eps

    def get_learning_rate(self):
        return self._learning_rate

    def set_learning_rate(self, lr):
        self._learning_rate = lr

    def get_temperature(self):
        return self._temperature

    def set_temperature(self, temp):
        self._temperature = temp

    def is_softmax_action(self):
        return self._softmax_action

    def set_softmax_action(self, is_softmax_action):
        self._softmax_action = is_softmax_action

    def update(self):
        if len(self._prediction_history) > 1:
            error = self._prediction_history[-1] - self._prediction_history[-2]
            sum_grad_W_in = np.sum(self._prediction_gradient_history[:-1], axis=0)[0]
            sum_grad_W_out = np.sum(self._prediction_gradient_history[:-1], axis=0)[1]
            self._sess.run([self._update_W_in, self._update_W_out],
                           feed_dict={self._lr: self._learning_rate, self._error: error,
                                      self._sum_grad_W_in: sum_grad_W_in, self._sum_grad_W_out: sum_grad_W_out})

    def update_terminal(self, target):
        error = target - self._prediction_history[-1]
        sum_grad_W_in = np.sum(self._prediction_gradient_history, axis=0)[0]
        sum_grad_W_out = np.sum(self._prediction_gradient_history, axis=0)[1]
        self._sess.run([self._update_W_in, self._update_W_out],
                       feed_dict={self._lr: self._learning_rate, self._error: error,
                                  self._sum_grad_W_in: sum_grad_W_in, self._sum_grad_W_out: sum_grad_W_out})

    def update_offline(self, target):
        if len(self._prediction_history) > 0:
            for i in range(1, len(self._prediction_history) + 1):
                prev = self._prediction_history[i - 1]
                last = self._prediction_history[i] if i < len(self._prediction_history) else target
                error = last - prev
                sum_grad_W_in = np.sum(self._prediction_gradient_history[:i], axis=0)[0]
                sum_grad_W_out = np.sum(self._prediction_gradient_history[:i], axis=0)[1]
                self._sess.run([self._update_W_in, self._update_W_out],
                               feed_dict={self._lr: self._learning_rate, self._error: error,
                                          self._sum_grad_W_in: sum_grad_W_in, self._sum_grad_W_out: sum_grad_W_out})

    def print_params(self):
        params = self._sess.run([self._W_in])
        print("W_in: %s" % params[0].tolist())
        params = self._sess.run([self._W_out])
        print("W_out: %s" % params[0].tolist())

    def print_gradients(self):
        print(self._prediction_gradient_history)

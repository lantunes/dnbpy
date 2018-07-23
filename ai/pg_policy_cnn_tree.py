import sys
sys.path.append("/Users/u6042446/Desktop/LuisRL/dnbpy/")
from ai import *
from dnbpy import *
import tensorflow as tf
import numpy as np
from Tree_cnn import *
import os


class PGPolicyCNN2Tree(Policy):
    """
    Adds a second convolutional layer.
    """
    def __init__(self, board_size, batch_size=1):
        #self._sess = tf.Session()
        self._board_size_tree_cnn = board_size
        self._board_size = (2,2)
        self._batch_size = batch_size

        edge_matrix = init_edge_matrix(self._board_size_tree_cnn)
        edge_matrix_base = init_edge_matrix(self._board_size)
        self._n_input_rows_tree_cnn = edge_matrix.shape[0]
        self._n_input_cols_tree_cnn = edge_matrix.shape[1]

        self._n_input_rows_base_cnn = edge_matrix_base.shape[0]
        self._n_input_cols_base_cnn = edge_matrix_base.shape[1]

        self._n_hidden = 300
        self._n_hidden_tree_cnn = 300
        self._n_filters_layer1_tree_cnn = 12
        self._n_filters_layer2_tree_cnn = 24

        self._n_kernel_size_layer1_tree_cnn = [2,2]
        self._n_kernel_size_layer2_tree_cnn = [2,2]

        self._n_output = len(init_board_state(board_size))
        self.init_base_CNN_graph()
        self.init_tree_CNN_graph()
        #Write another function for tree_CNN


    def init_tree_CNN_graph(self):
        """
        :Initializes Graph for tree_CNN
        :return:
        """
        self.graph_tree_cnn = tf.Graph()
        with self.graph_tree_cnn.as_default():
            #Input: an N by 4 matrix where N denotes the size of embedding
            self._input_tree_cnn = tf.placeholder("float", [None, self._n_hidden, 4], name="input_tree_cnn")
            num_leaves = np.power(4,(self._board_size_tree_cnn[0]-self._board_size[0]))

            self._input_tree_cnn_for_test = [tf.placeholder("float", [None, self._n_hidden,1], name="input_tree_cnn") for n in range(0,num_leaves)]
            self._input_reshaped_tree_cnn = tf.reshape(self._input_tree_cnn,
                                              shape=[tf.shape(self._input_tree_cnn)[0], self._n_hidden, 4,
                                                     1])
            self._action_taken_tree_cnn = tf.placeholder("float", [None, self._n_output], name="action_taken_tree_cnn")
            self._outcome = tf.placeholder(tf.float32, (None, 1), name="outcome_tree_cnn")
            self._lr = tf.placeholder("float", shape=[], name="learning_rate")

            #dimension of flattened layer: number_of_filters*(dim_x_input - kernel_size)*(dim_y_input-kernel_size) because there is no padding
            self.dim_flatten_layer = (self._n_hidden-self._n_kernel_size_layer2_tree_cnn[0]+1)*(4-self._n_kernel_size_layer2_tree_cnn[1]+1) * 24

            self._W_in_tree_cnn = tf.Variable(tf.random_normal
                                              ([self.dim_flatten_layer, self._n_hidden_tree_cnn], 0.0, 0.1), name="W_in_tree_cnn")

            self._b_in_tree_cnn = tf.Variable(tf.zeros([self._n_hidden_tree_cnn]), name="b_in_tree_cnn")
            self._W_out_tree_cnn = tf.Variable(tf.random_normal([self._n_hidden_tree_cnn, self._n_output], 0.0, 0.1), name="W_out_tree_cnn")

            #First conv layer: applies to an N by 4 matrix of embedding with N being the size of embedding
            self._conv_tree_cnn = tf.layers.conv2d(
                inputs=self._input_reshaped_tree_cnn,
                filters=self._n_filters_layer1_tree_cnn,
                kernel_size=self._n_kernel_size_layer1_tree_cnn,
                strides=(1, 1),
                padding="same",
                kernel_initializer=tf.random_normal_initializer(0.0, 0.1),
                activation=tf.nn.relu)

            #Second conv layer:
            self._conv2_tree_cnn = tf.layers.conv2d(
                inputs=self._conv_tree_cnn,
                filters=self._n_filters_layer2_tree_cnn,
                kernel_size=self._n_kernel_size_layer2_tree_cnn,
                strides=(1, 1),
                kernel_initializer=tf.random_normal_initializer(0.0, 0.1),
                activation=tf.nn.relu)

            #self._conv_flat_tree_cnn = tf.reshape(self._conv2_tree_cnn,
             #                                     [tf.shape(self._input_tree_cnn)[0], self.dim_flatten_layer])

            #dense_layer = tf.nn.tanh(tf.matmul(self._conv_flat_tree_cnn, self._W_in_tree_cnn) + self._b_in_tree_cnn)
            #self._embed_tree_cnn = dense_layer

            self._action_probs_tree_cnn = tf.nn.softmax(tf.matmul(dense_layer, self._W_out_tree_cnn))

            self._gen_embedding = self.gen_embbedding(self._input_tree_cnn_for_test)


            self._sess_tree_cnn = tf.Session(graph=self.graph_tree_cnn)
            self._sess_tree_cnn.run(tf.global_variables_initializer())




    def gen_embbedding(self,x):
        """

        :param x: List of base embeddings
        :return:
        """
        temp_embed_holder = []
        while len(x)>0:
            #node_embed = tf.reshape(x[0], np.shape(tree_embed_queue[0]) + (1,))
            node_embed = x[0]
            x = x[1:] #remove the current node
            temp_embed_holder.append(node_embed)
            if len(temp_embed_holder) == 4:
                input_to_parent = tf.concat(temp_embed_holder, axis=2) #form a size_embed
                #self._input_tree_cnn = input_to_parent
                #generate embbedding based on the tree-cnn model
                conv_flat_tree_cnn = tf.reshape(self._conv2_tree_cnn,
                                                      [tf.shape(input_to_parent)[0], self.dim_flatten_layer])
                dense_layer = tf.nn.tanh(tf.matmul(conv_flat_tree_cnn, self._W_in_tree_cnn) + self._b_in_tree_cnn)

                #out = self.get_tree_cnn_embed(input_to_parent)

                #parent_cnn_embediding = self._embed_tree_cnn
                #tree_embed_queue = tree_embed_queue + [parent_cnn_embediding]
                #temp_embed_holder = []

        return (dense_layer)


    def init_base_CNN_graph(self):
        self.graph_base_cnn = tf.Graph()
        with self.graph_base_cnn.as_default():
            self._input = tf.placeholder("float", [None, self._n_input_rows_base_cnn, self._n_input_cols_base_cnn], name="input")
            self._action_taken = tf.placeholder("float", [None, self._n_output], name="action_taken")
            #self._outcome = tf.placeholder(tf.float32, (None, 1), name="outcome")
            #self._lr = tf.placeholder("float", shape=[], name="learning_rate")

            self._W_in = tf.Variable(tf.random_normal([3 * 3 * 24, self._n_hidden], 0.0, 0.1), name="W_in")
            self._b_in = tf.Variable(tf.zeros([self._n_hidden]), name="b_in")
            self._W_out = tf.Variable(tf.random_normal([self._n_hidden, self._n_output], 0.0, 0.1), name="W_out")

            self._input_reshaped = tf.reshape(self._input,
                                              shape=[tf.shape(self._input)[0], self._n_input_rows_base_cnn, self._n_input_cols_base_cnn,
                                                     1])

            # Convolutional Layer 1
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

            # Convolutional Layer 2
            # Computes 24 features using a 3x3 filter with ReLU activation.
            # No padding is added.
            # Input Tensor Shape: [None, 5, 5, 12]
            # Output Tensor Shape: [None, 3, 3, 24]
            self._conv2 = tf.layers.conv2d(
                inputs=self._conv,
                filters=24,
                kernel_size=[3, 3],
                strides=(1, 1),
                kernel_initializer=tf.random_normal_initializer(0.0, 0.1),
                activation=tf.nn.relu)

            self._conv_flat = tf.reshape(self._conv2, [tf.shape(self._input)[0], 3 * 3 * 24])

            dense_layer = tf.nn.tanh(tf.matmul(self._conv_flat, self._W_in) + self._b_in)

            self._action_probs = tf.nn.softmax(tf.matmul(dense_layer, self._W_out))
            self._embed = dense_layer



            #self._cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
             #   logits=tf.matmul(dense_layer, self._W_out), labels=self._action_taken)
            #self._loss = self._cross_entropy * self._outcome
            #if self._batch_size > 1:
             #   self._loss = tf.reduce_mean(self._loss)
            #self._train_op = tf.train.GradientDescentOptimizer(self._lr).minimize(self._loss)

            #self._conv2d_kernel = [v for v in tf.global_variables() if v.name == 'conv2d/kernel:0'][0]
            #self._conv2d_bias = [v for v in tf.global_variables() if v.name == 'conv2d/bias:0'][0]
            #self._conv2d_kernel2 = [v for v in tf.global_variables() if v.name == 'conv2d_1/kernel:0'][0]
            #self._conv2d_bias2 = [v for v in tf.global_variables() if v.name == 'conv2d_1/bias:0'][0]

            self._sess = tf.Session(graph=self.graph_base_cnn)
            self._sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()


    def store_model(self,path,model_name):
        """
        :Stores the whole model in the given path
        :param path:
        :param model_name:
        :return:
        """
        save_path = self.saver.save(self._sess,os.path.join(path, model_name))

    def restore_base_model(self, path):
        """
        :This restores the base CNN model
        :param path: The path to the base-CNN model
        :return:
        """
        with self.graph_base_cnn.as_default():
            self.saver.restore(self._sess, path)

    def get_architecture(self):
        return "5x5-conv(3x3, relu, 12)-conv(3x3, relu, 24)-tanh(300)-softmax(1)"

    def get_base_cnn_embed(self,input):
        """
        Generates embedding for the base-CNN model
        :param input: edge-matrix for the base-board
        :return: embedding of the board
        """
        base_cnn_embedding = self._sess.run(self._embed, feed_dict={self._input: input})
        return (base_cnn_embedding)

    def get_tree_cnn_embed(self,input):

        tree_cnn_embedding = self._sess_tree_cnn.run(self._embed_tree_cnn,feed_dict = {self._input_tree_cnn: input})
        return (tree_cnn_embedding)

    def select_edge(self, board_state):
        """
        :Selects a new edge based on both treeCNN and baseCNN models
        :param board_state: the current string state of the board
        :return:
        """

        #Convert board_state to the edge-matrix
        edge_matrix = convert_board_state_to_edge_matrix(self._board_size_tree_cnn, board_state)
        #Extract all the leaves of the tree (all 2 by 2 sub-blocks)
        save_all_ranges = [[[0, self._board_size_tree_cnn[0]], [0, self._board_size_tree_cnn[0]]]]
        save_all_ranges = gen_sub_boards(save_all_ranges)
        #save_all_range: a list of indexed where each entry denoted a 2 by 2 sub-block of the baord

        base_embed_holder = [] #to store embedding for base-CNN
        tree_embed_queue = [] #to store embedding for tree-CNN


        #Loop over all the leaves and generate embedding for the first-level parent
        for sub_slice in save_all_ranges:
            sub_edge_matrix = np.array(get_sub_edge_matrix(np.array(edge_matrix),sub_slice))
            #Calculate embedding for base CNN
            sub_edge_matrix = np.reshape(sub_edge_matrix,(1,)+np.shape(sub_edge_matrix))
            base_cnn_embedding = self.get_base_cnn_embed(np.array(sub_edge_matrix))
            #sys.exit(1)
            base_cnn_embedding = np.reshape(base_cnn_embedding,np.shape(base_cnn_embedding)+(1,))
            base_embed_holder.append(base_cnn_embedding)

            """
            if len(base_embed_holder)==4:
                #Convert this to a form that can be consumed by CNN
                input_to_parent = np.concatenate(base_embed_holder,axis=2)
                #Generate embedding for the parent
                parent_cnn_embediding = self.get_tree_cnn_embed(input_to_parent)
                #Use Tree-CNN to map to another embedding
                tree_embed_queue.append(parent_cnn_embediding)
                base_embed_holder = []
            """

        #Bottom-up traversal in the created embedding (This will go up to the tree till getting to the root)
        #shape = np.shape(base_embed_holder)
        #base_embed_holder = np.reshape(base_embed_holder,shape[0:3])

        key_value_list = {key: value for (key, value) in zip(self._input_tree_cnn_for_test, base_embed_holder)}
        #print(key_value_list)
        test_embed = self._sess_tree_cnn.run(self._gen_embedding,feed_dict = dict(key_value_list))
        print(np.shape(test_embed))
        sys.exit(1)
        temp_embed_holder = []
        while len(tree_embed_queue)>0:
            node_embed = np.reshape(tree_embed_queue[0],np.shape(tree_embed_queue[0])+(1,))
            tree_embed_queue  = tree_embed_queue[1:]
            temp_embed_holder.append(node_embed)
            if len(temp_embed_holder)==4:
                input_to_parent = np.concatenate(temp_embed_holder,axis=2)
                parent_cnn_embediding = self.get_tree_cnn_embed(input_to_parent)
                tree_embed_queue = tree_embed_queue + [parent_cnn_embediding]
                temp_embed_holder = []

        root_embedding = node_embed[:,:,0] #This is the final embedding (the embedding of root)

        action_probs = self._sess_tree_cnn.run([self._action_probs_tree_cnn], feed_dict={self._input: [root_embedding]})
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
        params = self._sess.run([self._conv2d_kernel2])
        f.write("conv2d_kernel2: %s\n" % params[0].tolist())
        params = self._sess.run([self._conv2d_bias2])
        f.write("conv2d_bias2: %s\n" % params[0].tolist())

    def print_input(self,input):
        base_cnn_embedding = self._sess.run(self._input, feed_dict={self._input: input})
        return (base_cnn_embedding)

if __name__=="__main__":
    pass

from enum import Enum


class Type(Enum):
    Relu = 1
    Sigmoidal = 2
    Tanh = 3

class training_params():
    def __init__(self,rows,cols):

        self.num_folds = 10
        self.train_ratio = .9
        self.num_epochs = 50

        self.rows = rows
        self.cols = cols
        self.learning_rate = 0.001
        self.num_steps = 200
        self.batch_size = 128
        self.display_step = 10

        # Network Parameters
        self.num_input = (rows + 1) * (cols + 1)  # Investigate this
        self.num_outputs = (2 * rows * cols) + rows + cols  # dimension of actions (number of edges)
        self.dropout = 0.75  # Dropout, probability to keep units


class mlp_params(training_params):
    def __init__(self,rows,cols,hidden_units):
        training_params.__init__(self,rows,cols)
        self.num_input = (2 * rows * cols) + rows + cols
        self.dim_state = self.num_input
        self.layers = []

        num_inputs = self.num_input
        for h in hidden_units:
            layer = perceptron_layer(num_inputs,h,Type.Tanh)
            self.layers.append(layer)
            num_inputs = h

        #construct output-layer too
        layer = perceptron_layer(num_inputs,self.num_outputs,Type.Tanh)
        self.layers.append(layer)


class conv_params(training_params):
    def __init__(self,rows,cols):
        training_params.__init__(self,rows,cols)
        self.layers = []
        #Initialize layers
        #self.num_conv_layers = 1

        #Initializing an array of layer objects
        layer1_kernel_size = 5
        layer1_filter_size = 32
        layer1_pool_size = 2
        layer1 = conv_layer(self.num_input,int((self.num_input)/layer1_pool_size**2),layer1_kernel_size,layer1_filter_size,layer1_pool_size)
        self.layers.append(layer1)

        #Demo,intermediate layer
        layer2_kernel_size = 3
        layer2_filter_size = 64
        layer2_pool_size = 2
        layer2_input_dim = self.layers[-1].dim_output
        layer2 = conv_layer(layer2_input_dim,int(layer2_input_dim/(layer2_pool_size)**2),layer2_kernel_size,layer2_filter_size,layer2_pool_size)
        self.layers.append(layer2)

        #Fully-connected layer
        num_input = self.layers[-1].dim_output*self.layers[-1].filter_size
        fully_connected_dim = 1024
        layer_out = conv_layer(num_input,fully_connected_dim,-1,-1,-1)
        self.layers.append(layer_out)


class conv_layer:
    def __init__(self,dim_input,dim_output,kernel_size,filter_size,max_pool_size):
        self.dim_input = dim_input
        self.kernel_size = kernel_size
        self.filter_size = filter_size
        self.max_pool_size = max_pool_size
        self.dim_output = dim_output

class perceptron_layer:
    def __init__(self,dim_input,num_hidden_units,hidden_unit_nonlinearity):
        self.dim_input = dim_input
        self.dim_output = num_hidden_units
        self.num_hidden_units = num_hidden_units
        self.nonlinearity_type = hidden_unit_nonlinearity

if __name__=="__main__":
    #cnv = conv_params(27,27)
    mll = mlp_params(3,3,[100,200])


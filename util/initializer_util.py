import tensorflow as tf
import ast


class NumpyList(tf.keras.initializers.Initializer):
    def __init__(self, np_list, dtype=tf.float32):
        self.np_list = np_list
        self.dtype = dtype

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        return tf.convert_to_tensor(self.np_list, dtype=dtype)


def np_list_initializer(np_list):
    return NumpyList(np_list=np_list)


def existing_param_initializer(existing_params, param_name, default_val):
    if existing_params is not None:
        return np_list_initializer(existing_params[param_name])
    else:
        return default_val


def read_params(file_path):
    with open(file_path) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    # content = [print(x.strip()) for x in content]
    params = {}
    for line in content:
        line = line.strip()
        param_name, value_string = line.split(": ")
        params[param_name] = ast.literal_eval(value_string)
    return params


# import numpy as np
# print(np_list_initializer(np.array([0., 0.]))(shape=[1, 2]))

# params = read_params('/Users/u6046782/dnbpy/resources/cnn2_vs_L0_L1_L2_batch_01.txt')
# for k in params:
#     print(k)
#     print(params[k])

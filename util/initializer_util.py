import tensorflow as tf


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


# import numpy as np
# print(np_list_initializer(np.array([0., 0.]))(shape=[1, 2]))

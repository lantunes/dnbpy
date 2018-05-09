
import numpy as np
import random
import pandas as pd
train_data_index = 0
test_data_index = 0
full_data_index = 0



def reset_data_indexes():
    global train_data_index
    train_data_index = 0

def generate_batch_for_mlp_training(data_train_states,data_train_labels, batch_size,dim_state):
    """
    :param data_train:
    :param batch_size:
    :return:
    """
    global train_data_index

    min_index = train_data_index  # starting index
    max_index = min(train_data_index + batch_size - 1, len(data_train_states) - 1)

    state_batch = np.ndarray(shape=(max_index-min_index+1, dim_state), dtype=np.int32)
    labels = np.ndarray(shape=(max_index-min_index+1, dim_state), dtype=np.int32)

    index_counter = 0
    for index in range(min_index,max_index+1):

        binary_state_vector = []
        [binary_state_vector.append(int(x)) for x in data_train_states[index_counter]]
        multi_out_label = np.zeros([1,dim_state])
        multi_out_label[0,data_train_labels[index_counter]] = 1
        state_batch[index_counter,:] = binary_state_vector
        labels[index_counter,:] = multi_out_label
        index_counter+=1

    train_data_index = max_index+1
    return state_batch, labels

def generate_for_mlp_test(data_test_states,data_test_labels,dim_state):
    """
    :param data_train:
    :param batch_size:
    :return:
    """

    min_index = 0  # starting index
    max_index =len(data_test_states)-1

    state = np.ndarray(shape=(max_index-min_index+1, dim_state), dtype=np.int32)
    labels = np.ndarray(shape=(max_index-min_index+1, dim_state), dtype=np.int32)

    index_counter = 0
    for index in range(min_index,max_index+1):

        binary_state_vector = []
        [binary_state_vector.append(int(x)) for x in data_test_states[index_counter]]
        multi_out_label = np.zeros([1,dim_state])
        multi_out_label[0,data_test_labels[index_counter]] = 1
        state[index_counter,:] = binary_state_vector
        labels[index_counter,:] = multi_out_label
        index_counter+=1

    return state, labels


def n_fold_cross_validation(main_df, n_fold, list_of_indices):

    """

    :param main_df:
    :param n_fold:
    :param list_of_indices:
    :return:
    """
    all_indexes = np.array(range(0,len(list_of_indices)))
    fold_indexes = np.array_split(all_indexes, n_fold)  # generate different folds
    fold_train_df = []
    for x in fold_indexes:
        index_list = []
        [index_list.append(list_of_indices[y]) for y in x]  # generate list of training indices
        this_fold_df = main_df.loc[main_df['index'].isin(set(index_list))]
        fold_train_df.append(this_fold_df)
    return (fold_train_df)


def generate_train_validation_form_fold(fold_index, fold_train_df):
    """

    :param fold_index:
    :param fold_train_df:
    :return:
    """
    all_train_instances = fold_train_df[:fold_index] + fold_train_df[(fold_index + 1):]
    empty_df = pd.DataFrame()
    for x in all_train_instances:
        empty_df = empty_df.append(x)
    validation_instances = fold_train_df[fold_index]
    return (empty_df, validation_instances)
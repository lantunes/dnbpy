from dnbpy import *


def print_info(board_size, policy, num_episodes, learning_rate=None, min_learning_rate=None, epsilon=None,
               min_epsilon=None, temperature=None, min_temperature=None, mode='self-play', reward='delayed',
               updates='online', architecture=None, decay_speed=None, rand_prob=None, batch_size=None,
               learning_rate_schedule=None, epsilon_schedule=None, temperature_schedule=None,
               dropout_keep_prob=None, use_symmetries=None, num_episodes_per_policy_update=None,
               num_episodes_per_opponent_cache=None, opponent_pool_max_size=None, episodes_per_thread=None,
               mcts=None, mcts_simulations=None, mcts_c=None):
    print("board size: %sx%s" % board_size)
    print("policy: %s" % policy.__class__.__name__)
    if mcts is not None:
        print("MCTS policy: %s" % mcts.__class__.__name__)
    if mcts_simulations is not None:
        print("MCTS simulations: %s" % mcts_simulations)
    if mcts_c is not None:
        print("MCTS c: %s" % mcts_c)
    if architecture is not None:
        print("architecture: %s" % architecture)
    print("mode: %s" % mode)
    print("reward: %s" % reward)
    print("updates: %s" % updates)
    print("num episodes: %s" % num_episodes)
    if learning_rate is not None:
        print("learning rate: %s" % learning_rate)
    if min_learning_rate is not None:
        print("min. learning rate: %s" % min_learning_rate)
    if epsilon is not None:
        print("epsilon: %s" % epsilon)
        print("min. epsilon: %s" % min_epsilon)
    if temperature is not None:
        print("temperature: %s" % temperature)
        print("min. temperature: %s" % min_temperature)
    if rand_prob is not None:
        print("random player prob.: %s" % rand_prob)
    if batch_size is not None:
        print("batch size: %s" % batch_size)
    if decay_speed is not None:
        print("decay speed: %s" % decay_speed)
    if learning_rate_schedule is not None:
        print("learning rate schedule: %s" % sorted(learning_rate_schedule.items()))
    if epsilon_schedule is not None:
        print("epsilon schedule: %s" % sorted(epsilon_schedule.items()))
    if temperature_schedule is not None:
        print("temperature schedule: %s" % sorted(temperature_schedule.items()))
    if dropout_keep_prob is not None:
        print("dropout keep prob.: %s" % dropout_keep_prob)
    if use_symmetries is not None:
        print("transitions to symmetries: %s" % use_symmetries)
    if num_episodes_per_policy_update is not None:
        print("params updated every %s episodes" % num_episodes_per_policy_update)
    if num_episodes_per_opponent_cache is not None:
        print("current policy added to pool every %s episodes" % num_episodes_per_opponent_cache)
    if opponent_pool_max_size is not None:
        print("max opponent pool size: %s" % opponent_pool_max_size)
    if episodes_per_thread is not None:
        print("# episodes per thread: %s" % episodes_per_thread)


def to_one_hot_action(board_state, edge_index):
    action_vector = [0]*len(board_state)
    action_vector[edge_index] = 1
    return action_vector


def to_state_action_pair_symmetries(board_size, state, action):
    #state_edge_matrix = convert_board_state_to_edge_matrix(board_size, state)
    state_edge_matrix = state[0,:,:,0]
    state_box_matrix = state[0,:,:,1]
    action_edge_matrix = convert_board_state_to_edge_matrix(board_size, action)
    # I symmetry
    state_i = np.array(state_edge_matrix)
    action_i = np.array(action_edge_matrix)
    rows = state_i.shape[0]
    cols = state_i.shape[1]

    if rows == cols:
        # S symmetry
        state_s = np.flip(state_i, 1)
        action_s = np.flip(action_i, 1)
        box_s = np.flip(state_box_matrix,1)
        new_state_s = np.concatenate([np.reshape(state_s,(1,)+np.shape(state_s)+(1,)),np.reshape(box_s,(1,)+np.shape(box_s)+(1,))],axis=3)

        # R symmetry
        state_r = np.rot90(state_i, 1)
        action_r = np.rot90(action_i, 1)
        box_r = np.rot90(state_box_matrix,1)
        new_state_r = np.concatenate([np.reshape(state_r, (1,) + np.shape(state_r) + (1,)),
                       np.reshape(box_r, (1,) + np.shape(box_r) + (1,))],axis=3)


        # R^2 symmetry
        state_r2 = np.rot90(state_i, 2)
        action_r2 = np.rot90(action_i, 2)
        box_r2 = np.rot90(state_box_matrix,2)
        new_state_r2 = np.concatenate([np.reshape(state_r2, (1,) + np.shape(state_r2) + (1,)),
                       np.reshape(box_r2, (1,) + np.shape(box_r2) + (1,))],axis=3)

        # R^3 symmetry
        state_r3 = np.rot90(state_i, 3)
        action_r3 = np.rot90(action_i, 3)
        box_r3 = np.rot90(state_box_matrix,3)
        new_state_r3 = np.concatenate([np.reshape(state_r3, (1,) + np.shape(state_r3) + (1,)),
                       np.reshape(box_r3, (1,) + np.shape(box_r3) + (1,))],axis=3)

        # SR symmetry
        state_sr = np.rot90(state_s, 1)
        action_sr = np.rot90(action_s, 1)
        box_sr = np.rot90(box_s,1)
        new_state_sr = np.concatenate([np.reshape(state_sr, (1,) + np.shape(state_sr) + (1,)),
                       np.reshape(box_sr, (1,) + np.shape(box_sr) + (1,))],axis=3)

        # SR^2 symmetry
        state_sr2 = np.rot90(state_s, 2)
        action_sr2 = np.rot90(action_s, 2)
        box_sr2 = np.rot90(box_s,2)
        new_state_sr2 = np.concatenate([np.reshape(state_sr2, (1,) + np.shape(state_sr2) + (1,)),
                       np.reshape(box_sr2, (1,) + np.shape(box_sr2) + (1,))],axis=3)

        # SR^3 symmetry
        state_sr3 = np.rot90(state_s, 3)
        action_sr3 = np.rot90(action_s, 3)
        box_sr3 = np.rot90(box_s,3)
        new_state_sr3 = np.concatenate([np.reshape(state_sr3, (1,) + np.shape(state_sr3) + (1,)),
                       np.reshape(box_sr3, (1,) + np.shape(box_sr3) + (1,))],axis=3)


        all_possible = [[state, action_i], [new_state_s, action_s], [new_state_r, action_r],
                        [new_state_r2, action_r2], [new_state_r3, action_r3], [new_state_sr, action_sr],
                        [new_state_sr2, action_sr2], [new_state_sr3, action_sr3]]

    else:
        # S symmetry
        state_s = np.flip(state_i, 1)
        action_s = np.flip(action_i, 1)

        # R^2 symmetry
        state_r2 = np.rot90(state_i, 2)
        action_r2 = np.rot90(action_i, 2)

        # SR^2 symmetry
        state_sr2 = np.rot90(state_s, 2)
        action_sr2 = np.rot90(action_s, 2)

        all_possible = [[state_i, action_i], [state_s, action_s], [state_r2, action_r2], [state_sr2, action_sr2]]

    def contains(list, arr):
        for item in list:
            if np.array_equal(item[0], arr[0]):
                return True
        return False

    symmetries = []
    for sym in all_possible:
        if not contains(symmetries, sym):
            symmetries.append([sym[0], convert_edge_matrix_to_board_state(sym[1])])

    return symmetries


def append_transitions(states, actions, outcomes, all_transitions, use_symmetries, board_size):
    for i, _ in enumerate(actions):
        state = states[i] #this is itself another list of states
        action = actions[i]
        reward = outcomes[i]
        if use_symmetries:
            state_action_symmetries = to_state_action_pair_symmetries(board_size, state, action)
            for symmetry in state_action_symmetries:
                all_transitions.append([symmetry[0], symmetry[1], reward])
        else:
            all_transitions.append([state, action, reward])

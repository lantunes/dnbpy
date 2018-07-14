

def print_info(board_size, policy, num_episodes, learning_rate=None, min_learning_rate=None, epsilon=None,
               min_epsilon=None, temperature=None, min_temperature=None, mode='self-play', reward='delayed',
               updates='online', architecture=None, decay_speed=None, rand_prob=None, batch_size=None,
               learning_rate_schedule=None, epsilon_schedule=None, temperature_schedule=None):
    print("board size: %sx%s" % board_size)
    print("policy: %s" % policy.__class__.__name__)
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


def to_one_hot_action(board_state, edge_index):
    action_vector = [0]*len(board_state)
    action_vector[edge_index] = 1
    return action_vector

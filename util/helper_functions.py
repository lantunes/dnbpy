

def print_info(board_size, num_episodes, learning_rate, min_learning_rate, epsilon, min_epsilon, policy):
    print("board size: %sx%s" % board_size)
    print("num episodes: %s" % num_episodes)
    print("learning rate: %s" % learning_rate)
    print("min. learning rate: %s" % min_learning_rate)
    print("epsilon: %s" % epsilon)
    print("min. epsilon: %s" % min_epsilon)
    print("policy: %s" % policy.__class__.__name__)
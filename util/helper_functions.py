

def print_info(board_size, policy, num_episodes, learning_rate, min_learning_rate, epsilon=None, min_epsilon=None,
               temperature=None, min_temperature=None, mode='self-play', reward='delayed', updates='online',
               architecture=None):
    print("board size: %sx%s," % board_size)
    print("policy: %s," % policy.__class__.__name__)
    if architecture is not None:
        print("architecture: %s," % architecture)
    print("mode: %s," % mode)
    print("reward: %s," % reward)
    print("updates: %s," % updates)
    print("num episodes: %s," % num_episodes)
    print("learning rate: %s," % learning_rate)
    print("min. learning rate: %s," % min_learning_rate)
    if epsilon is not None:
        print("epsilon: %s," % epsilon)
        print("min. epsilon: %s," % min_epsilon)
    else:
        print("temperature: %s," % temperature)
        print("min. temperature: %s" % min_temperature)

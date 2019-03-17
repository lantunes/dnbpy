import numpy as np
# import matplotlib.pyplot as plt
# import pylab


def gen_rate_exponential(n, max, min, n_max, speed=1):
    n_max /= speed
    if n > n_max:
        return min
    alpha = 2 * max
    beta = np.log(alpha / min - 1) / n_max
    return alpha / (1 + np.exp((beta * n)))


def gen_rate_step(n, schedule):
    """
    :param n: the current iteration number 
    :param schedule: a dictionary where the keys are the min value for the step and the values are the corresponding 
                     rate; for example, {0: 0.005, 200: 0.0002, 400: 0.0001} is an annealing schedule where iterations
                     0 to 200 get a rate of 0.005, iterations 201 to 400 get a rate of 0.0002, and iterations >400 get
                     a rate of 0.0001; importantly, schedule dictionaries will be ordered by key, and the first key 
                     must be 0 
    :return: the corresponding rate for the iteration number n
    """
    sorted_keys = sorted(schedule.keys())
    if len(sorted_keys) < 1 or sorted_keys[0] != 0:
        raise Exception("the schedule must contain 0 as a key")
    for k in reversed(sorted_keys):
        if n > k:
            return schedule[k]
        if n == 0:
            return schedule[0]


# def plot_rate(max, min, num_episodes, speed, fn):
#     episodes = []
#     vals = []
#     for episode_num in range(1, num_episodes + 1):
#         val = fn(episode_num, max, min, num_episodes, speed)
#         vals.append(val)
#         episodes.append(episode_num)
#         print(val)
#     pylab.plot(episodes, vals, 'k-', linewidth=0.5, marker='.', markersize=2)
#     plt.xlabel("iteration", size=15)
#     plt.ylabel("rate", size=15)
#     plt.grid(True, linewidth=0.2)
#     plt.show()


# plot_rate(1, 0.01, 100000, 1.17, gen_rate_exponential)

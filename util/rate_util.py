import numpy as np
import matplotlib.pyplot as plt
import pylab


def gen_rate_exponential(n, max, min, n_max, speed=1):
    n_max /= speed
    if n > n_max:
        return min
    alpha = 2 * max
    beta = np.log(alpha / min - 1) / n_max
    return alpha / (1 + np.exp((beta * n)))


def plot_rate(max, min, num_episodes, speed, fn):
    episodes = []
    vals = []
    for episode_num in range(1, num_episodes + 1):
        val = fn(episode_num, max, min, num_episodes, speed)
        vals.append(val)
        episodes.append(episode_num)
        print(val)
    pylab.plot(episodes, vals, 'k-', linewidth=0.5, marker='.', markersize=2)
    plt.xlabel("iteration", size=15)
    plt.ylabel("rate", size=15)
    plt.grid(True, linewidth=0.2)
    plt.show()


# plot_rate(1, 0.01, 100000, 1.17, gen_rate_exponential)

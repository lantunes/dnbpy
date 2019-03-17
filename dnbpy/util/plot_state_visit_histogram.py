import matplotlib.pyplot as plt
import numpy as np

with open('../out/state_visitation_count_plot.txt', 'r') as f:
    data = eval(f.read())

count  = list(zip(*data))[1]
# x_pos = np.arange(len(count))
# plt.bar(x_pos, count, align='center')
# plt.show()
print(np.mean(count))
print(np.std(count))
print(np.min(count))
print(np.max(count))
print(np.median(count))
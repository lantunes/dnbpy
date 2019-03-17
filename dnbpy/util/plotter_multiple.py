import matplotlib.pyplot as plt
import pylab

data = pylab.loadtxt('../out/plotdata.txt', delimiter=',', usecols=(0, 1, 2, 3))

pylab.plot(data[:, 0], data[:, 1], 'k-', linewidth=0.5, marker='.', markersize=2, label='L0', color='black')
pylab.plot(data[:, 0], data[:, 2], 'k-', linewidth=0.5, marker='.', markersize=2, label='L1', color='blue')
pylab.plot(data[:, 0], data[:, 3], 'k-', linewidth=0.5, marker='.', markersize=2, label='L2', color='red')

plt.xlabel("iteration", size=15)
plt.ylabel("performance", size=15)
plt.grid(True, linewidth=0.2)
plt.legend(loc='best')

plt.show()

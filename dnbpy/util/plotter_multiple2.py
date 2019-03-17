import matplotlib.pyplot as plt
import pylab

data = pylab.loadtxt('../out/plotdata.txt', delimiter=',', usecols=(0, 1, 2, 3, 4, 5, 6))

pylab.plot(data[:, 0], data[:, 1], linestyle='solid', linewidth=0.5, marker='.', markersize=2, label='L0', color='blue')
pylab.plot(data[:, 0], data[:, 2], linestyle='dashed', linewidth=0.5, marker='.', markersize=2, label='L1', color='blue')
pylab.plot(data[:, 0], data[:, 3], linestyle='dotted', linewidth=0.5, marker='.', markersize=2, label='L2', color='blue')

pylab.plot(data[:, 0], data[:, 4], linestyle='solid', linewidth=0.5, marker='.', markersize=2, label='L0-C', color='red')
pylab.plot(data[:, 0], data[:, 5], linestyle='dashed', linewidth=0.5, marker='.', markersize=2, label='L1-C', color='red')
pylab.plot(data[:, 0], data[:, 6], linestyle='dotted', linewidth=0.5, marker='.', markersize=2, label='L2-C', color='red')

plt.xlabel("iteration", size=15)
plt.ylabel("performance", size=15)
plt.grid(True, linewidth=0.2)
plt.legend(loc='best')

plt.show()

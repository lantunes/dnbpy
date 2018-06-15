import matplotlib.pyplot as plt
import pylab

data = pylab.loadtxt('../out/plotdata.txt', delimiter=',', usecols=(0, 1))

pylab.plot(data[:, 0], data[:, 1], 'k-', linewidth=0.5, marker='.', markersize=2)

plt.xlabel("iteration", size=15)
plt.ylabel("performance", size=15)
plt.grid(True, linewidth=0.2)

plt.show()

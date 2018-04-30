import matplotlib.pyplot as plt
import pylab

data = pylab.loadtxt('../out/plotdata.txt', delimiter=',', usecols=(0, 1))

pylab.plot(data[:, 0], data[:, 1], 'ko-', linewidth=3)

plt.xlabel("iteration", size=15)
plt.ylabel("reward", size=15)
plt.grid(True)

plt.show()

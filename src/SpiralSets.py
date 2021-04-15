# The point of this entire file is so we can use it to create random sets of nonlinear data
import random
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

samp = random.randint(50,100)
numclass = random.randint(5,10)
nnfs.init()

x, y = spiral_data(samples=samp, classes=numclass)

plt.scatter(x[:, 0], x[:, 1])
plt.show()
print("I hate poeple in general")
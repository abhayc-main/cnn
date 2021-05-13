# The point of this entire file is so we can use it to create random sets of nonlinear data
import random
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
"""
samp = random.randint(50,100)
numclass = random.randint(5,10)
"""
nnfs.init()

X, y = spiral_data(samples=100, classes=3)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
plt.show()
print("I hate poeple in general")




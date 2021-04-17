#The simplest activation function is a step function. 
# In a single neuron, if the weights · inputs + bias results in a value greater than 0, the neuron will fire and output a one, otherwise, it will output a 0.
# Pretty Trash function NGL
# Rarelly used adn we are only learning about to get an idea.
# It's a linear function aka HOT GARBAGE.

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

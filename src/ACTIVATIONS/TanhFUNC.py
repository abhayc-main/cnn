# The tanh function is considered more of an practice fucntion.
# YOu have to do the sigmoid to do this (KINDA DUMB)
# Sounds cool...

"""
It takes a real value as input and squashes it in the range(-1, 1).

In practice, the tanh activation is preferred over the sigmoid activation.
It is also common to use the tanh function in state to state transition models(recurrent neural networks).
The tanh function also suffers from the gradient saturation problem and kills gradients when saturated.
"""
import numpy as np
import random
# WHen we import something as a name we are calling it sum
inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]
# Biases are a vector Quantity because it's a 1D array.

weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]


# y=f(∑i=1nwixi+b)
# ^^^ Dot product formula
# THE DOT PRODUCT IS JUST THE PRODUCT OF 2 VECTORS
# The function numpy. dot() in Python returns a Dot product of two arrays x and y.
#Basically since Weights is a Matric it's gonan do the dot product of each vector.
# IT RETURNS AN ARRAY OF DOT  PRODUCTS
output = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(output, np.array(weights2).T) + biases2
print(output)
print("      ")
print(layer2_outputs)
# .T is the transposing function and the np.array creates an array
# WHen transposing we can choose either one to swtich the R and C
# WHen we pass the weights and the inputs to numpy, arrays for the weights and the inputs are being created. <BackENd>

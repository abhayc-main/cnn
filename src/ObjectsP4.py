import numpy as np
import random
# WHen we import something as a name we are calling it sum
# Capital x is just a common practice dw
np.random.seed(0)
X = [[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]] 
# The Self keyword just means that we can pass it in for a object.
#The __init__ BASICALLY IS JUST A CONSTRCUTOR KEYWORD.
# Look at __init__Self.py for
class LayerThick:
    def __init__(self, n_inputs, n_neurons):
        # Basically we are intializing out Inputs and Weights
        # We want the shape too.
        # WHen alayer is created we need 2 things...
        # WHat is the size of the inputs and how many neurons we want in it.
        self.weights = 0.1 * np.random.randn(n_inputs,n_neurons)
        #we multiplieing the inputs and neurons.
        self.biases = np.zeroes(1,n_neurons)
    def forward(self,inputs,weights):
        self.output = np.dot(inputs, self.weights) + self.biases
        

print(0.1*np.random.randn(4,3))


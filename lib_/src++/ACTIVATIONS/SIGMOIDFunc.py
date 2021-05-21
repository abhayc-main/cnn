# The Better function.
# The main problem with the STEP FUNCTION is...
# 1) It's linear
# 2) It's linear
# 3) It's linear
# But 4) It doesn't give the optimizer. (Tweaks Weights and Biases.) 
# (One of the main objects that train the network) a good amount of data for it to change the values.
# Its just liek a boolean: "YES" "NO" but in this situtation: "ON" or "OFF" 0 = "OFF" 1 = "ON"
# Basically it gives out less info.
# BUT>>> THE sigmoid function aint about that.
# Its more detailed and provides more info.
# FORMULA>>>>>>>>>^^^^>?>>> σ(x) = 1 ∕ (1 + e-x)

"""
The sigmoid function is not used any more. It has two major drawbacks –

VANISHING GRADIENT PROBLEM...
    The derivative of a sigmoid function is σ(x)(1 – σ(x)). During backpropagation, when the output of a neuron becomes 0 or 1, the gradient becomes 0. As a result the weights of the neuron do not get updated. These neurons are called saturated neurons. Not only this, the weight of the neurons connected to this saturated  neuron also get slowly updated. Hence, a network with sigmoid activation may not backpropagate if there are many saturated neurons present.

The exp() function is computationally expensive.
"""
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Dense layer
class Layer_Dense:

    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass

    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases


# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Perform a forward pass of our training data through this layer
dense1.forward(X)

# Let's see output of the first few samples:
print(dense1.output[:5])




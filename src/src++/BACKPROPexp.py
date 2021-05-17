# Chain Rule is kind of like the transitive property.
# Suppose we are a collecting the heights and the weights of someone.
# >> They would have a graph with a slope of 2/1 == 2. (For every 1 unit increased in weight, 2 units are increased into height.)
# SUppose we have another graph collecting the weights and shoe size.
# >> They would have another graph with a slope of (1/4). (For every 1 unit increased in height, 1/4 units are increased into shoe size.))
# So these two graphs are corresponding.
# Weight can predict height and height can predict shoe size.
# So we if we want to determine the exact changes with respect to changes in weight....
# WE can take the derivative of the equation if shoe size.....
#==================== (dshoesize/dweight) = (dheight/dweight) x (dshoesize/dheight)
# The relationship above is called the chain rule.

# Its used to help create a form of multiple function outputs and multiply them for the final outcome.

#VERY IMPORTANT-------------

"""
Derivatives of a function f(x):
    - how fast f changes around x (slope)
    - will f increase or decrease if we increse x
    - https://www.youtube.com/watch?v=3p09AyD-eWI


"""

#=============BACKPROPAGATION(With Basic levels of Optimization)===============#
# SUppose the network we are wokring it isnt trained (obviously).
# The network is supposes to recognize hand written images and We input a 2 but we get all sorts of wrong answers.
# We first find the loss of the network.
# Since we want the network to classify it as an image of a 2 but the output value of the "2" in the layer is low,....
# WE CAN NUDGE ALL THE weights AND BIASES UP FOR the "2" and nudge all the other neuron values down. And the the should be proportional to its size as well
#   (wrong answer: 6 ~ 0.9 >>> thsi should be nudged down more than the other ones because the network thinks this is the highest value.)

# WEIGHTS ONLY>>>>>
#   Once you go back you should retrace your steps to see which weights have the moist effect. And then you should change those.

# BIASES ONLY>>>>
#   Once you go back change the bias of the neuron in the prevoius layer that influences the Cost Function

# Basically you have a list of nudges to every neuron that you have to do.
# Then you recursivly apply the same properties to the first layer weights and biases.


# >>>>>>>>>>>>>> THE MAIN KEY IDEA IS THAT THIS WORKS FOR ONLY A 2 SO WE NEED TO DO THIS FOR ALL POSSIBLE OUTCOMES.
# Every SINGLE Training DATA needs to be back propagated.

# Forward pass
x = [1.0, -2.0, 3.0]  # input values
w = [-3.0, -1.0, 2.0]  # weights
b = 1.0  # bias
# Multiplying inputs by weights
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]
# Adding weighted inputs and a bias
z = xw0 + xw1 + xw2 + b
# ReLU activation function
y = max(z, 0)
# Backward pass
# The derivative from the next layer
dvalue = 1.0

# Derivative of ReLU and the chain rule
drelu_dz = dvalue * (1. if z > 0 else 0.)
print(drelu_dz)
# Partial derivatives of the multiplication, the chain rule
dsum_dxw0 = 1
dsum_dxw1 = 1
dsum_dxw2 = 1
dsum_db = 1
drelu_dxw0 = drelu_dz * dsum_dxw0
drelu_dxw1 = drelu_dz * dsum_dxw1
drelu_dxw2 = drelu_dz * dsum_dxw2
drelu_db = drelu_dz * dsum_db
print(drelu_dxw0, drelu_dxw1, drelu_dxw2, drelu_db)
# Partial derivatives of the multiplication, the chain rule
dmul_dx0 = w[0]
dmul_dx1 = w[1]
dmul_dx2 = w[2]
dmul_dw0 = x[0]
dmul_dw1 = x[1]
dmul_dw2 = x[2]
drelu_dx0 = drelu_dxw0 * dmul_dx0
drelu_dw0 = drelu_dxw0 * dmul_dw0
drelu_dx1 = drelu_dxw1 * dmul_dx1
drelu_dw1 = drelu_dxw1 * dmul_dw1
drelu_dx2 = drelu_dxw2 * dmul_dx2
drelu_dw2 = drelu_dxw2 * dmul_dw2
print(drelu_dx0, drelu_dw0, drelu_dx1, drelu_dw1, drelu_dx2, drelu_dw2)
#===========================================================
# This is cool and all but we need to do this more effecient
import numpy as np
# Passed in gradient from the next layer
# for the purpose of this example we're going to use
# an array of an incremental gradient values
dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.],
                    [3., 3., 3.]])
# We have 3 sets of inputs - samples
inputs = np.array([[1, 2, 3, 2.5],
                   [2., 5., -1., 2],
                   [-1.5, 2.7, 3.3, -0.8]])
# We have 3 sets of weights - one set for each neuron
# we have 4 inputs, thus 4 weights
# recall that we keep weights transposed
weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

# One bias for each neuron
# biases are the row vector with a shape (1, neurons)
biases = np.array([[2, 3, 0.5]])
# Forward pass
layer_outputs = np.dot(inputs, weights) + biases  # Dense layer
relu_outputs = np.maximum(0, layer_outputs)  # ReLU activation
# Let's optimize and test backpropagation here
# ReLU activation - simulates derivative with respect to input values
# from next layer passed to current layer during backpropagation
drelu = relu_outputs.copy()
drelu[layer_outputs <= 0] = 0
# Dense layer
# dinputs - multiply by weights
dinputs = np.dot(drelu, weights.T)
# dweights - multiply by inputs
dweights = np.dot(inputs.T, drelu)
# dbiases - sum values, do this over samples (first axis), keepdims
# since this by default will produce a plain list -
# we explained this in the chapter 4
dbiases = np.sum(drelu, axis=0, keepdims=True)
# Update parameters
weights += -0.001 * dweights
biases += -0.001 * dbiases
print(weights)
print(biases)

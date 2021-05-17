import numpy as np
import math
# The Softmax activation function.
# We will use this for the output layer.
# The main reason we need another function for the output layer is we need osmehting to create a probability to that the Network can make a prediction.
# We can make a prediction maybe correctly whenusing the ReLU and when we ahve all positive values but when we have a negative value, Relu clips negatives to zero.
# Since both neurons are negative the network outputs are both zero. SO it cant be make a prediction.
# Hence we need a function whos equation doesnt clip the output to a zero.
# YAYYYY softmax.
#  \sigma(\vec{z})_{i}=\frac{e^{z_{i}}}{\sum_{j=1}^{K} e^{z_{j}}}
# We also expionetiate the function so we can mkae them non negative.
# Then we get a very high number, so then we "normalize" them (divide them) so we get a vector of small values.
# THEN WE ADD IT UP and get hte final number f that neuron. 

# This Next Part is an example.

layer_outputs = [4.8, 1.21, 2.385]
E = math.e
exp_values = []
for output in layer_outputs:
    exp_values.append(E ** output)  # ** - power operator in Python
print('exponentiated values:')
print(exp_values)

# Normalizing...

norm_base = sum(exp_values)  # We sum all values
norm_values = []
for value in exp_values:
    norm_values.append(value / norm_base)
print('Normalized exponentiated values:')
print(norm_values)
print('Sum of normalized values:', sum(norm_values))

# ^^^^ above is on way of doing it without numpy
# WITH NUMPY +++====================

# Values from the earlier previous when we described
# what a neural network is
layer_outputs = [4.8, 1.21, 2.385]
# For each value in a vector, calculate the exponential value
exp_values = np.exp(layer_outputs)
print('exponentiated values:')
print(exp_values)
# Now normalize values
norm_values = exp_values / np.sum(exp_values)
print('normalized exponentiated values:')
print(norm_values)
print('sum of normalized values:', np.sum(norm_values))




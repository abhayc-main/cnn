import numpy as np

inputs = [1.0, 2.0, 3.0, 2.5]
# Inputs are a vector Quantity because it's a 1D array.

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
# The Weights is a Matrix because its a 2d array 

biases = [2.0, 3.0, 0.5]
# Biases are a vector Quantity because it's a 1D array.

# y=f(∑i=1nwixi+b)
# ^^^ Dot product formula
# THE DOT PRODUCT IS JUST THE PRODUCT OF 2 VECTORS 
# The function numpy. dot() in Python returns a Dot product of two arrays x and y.
#Basically since Weights is a Matric it's gonan do the dot product of each vector.
# IT RETURNS AN ARRAY OF DOT  PRODUCTS
output = np.dot(weights, inputs) + biases 
print(output)
import numpy as np
# WHen we import something as a name we are calling it sum
inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]
# Biases are a vector Quantity because it's a 1D array.

# y=f(∑i=1nwixi+b)
# ^^^ Dot product formula
# THE DOT PRODUCT IS JUST THE PRODUCT OF 2 VECTORS
# The function numpy. dot() in Python returns a Dot product of two arrays x and y.
#Basically since Weights is a Matric it's gonan do the dot product of each vector.
# IT RETURNS AN ARRAY OF DOT  PRODUCTS
output = np.dot(inputs, np.array(weights).T) + biases
# .T is the transposing function and the np.array creates an array
# WHen transposing we can choose either one to swtich the R and C 
# WHen we pass the weights and the inputs to numpy, arrays for the weights and the inputs are being created. <BackENd>
print(output)

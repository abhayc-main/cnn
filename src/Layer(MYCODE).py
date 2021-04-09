# Coding our first neuron
inputs = [1,2,3,2.5]
#these are exmaples of inputs meaning that they come from neurons of the Prevoius layer.
weights = [0.2,0.8,-0.5,1]
weights1 = [0.5, -0.91, -0.5, 0.26]
weights2 = [-0.26, -0.27, 0.17, 0.87]
biases = 2
biases1 = 3
biases2 = 0.5

output = 0
output2 = 0
output3 = 0
for i in range(len(inputs)):
    output+=inputs[i]*weights[i]
for i in range(len(inputs)):
    output += inputs[i]*weights1[i]
for i in range(len(inputs)):
    output += inputs[i]*weights2[i]
output+=biases
print(output)
 
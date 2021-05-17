"""
The ReLU is the most popular and commonly used activation function. It can be represented as –

f(x) = max(0, x)

It takes a real value as input. The output is x, when x > 0 and is 0, when x < 0. It is mostly preferred over sigmoid and tanh.

Advantages >>>>>>>>>>>>>>>>>>

Accelerates the convergence of SGD compared to sigmoid and tanh(around 6 times).
No expensive operations are required.

Disadvantages >>>>>>>>>>>>>>>

It is fragile and might die during training. If the learning rate is too high the weights may change to a value that causes the neuron to not get updated at any data point again.
"""
# Debating against the ReLU and the Leaky ReLU

# There are some interesting complications to the ReLU when it comes to optimizing weights and biases
# for ReLU...
# >>> IF the weigth and the bias are both = to 0 then the ReLU on a graph will be a straight line. The input can be 4 million but it doenst matter because the biases and the weights prevent it.
# >>>> If the weight is above 0 and the bias remains equal to 0, the steepness of the line on the graph will change. But the bend will still start at 0 because our bias is 0.
# >>>>> Now if the weight remains the same (1.00) but we change the bias to be 0.50 then activation of the function will happen sooner.
"""
 >>>>>> With a negative weight and this single neuron, the function has become a question of when this
neuron deactivates. Up to this point, you’ve seen how we can use the bias to offset the function
horizontally, and the weight to influence the slope of the activation. Moreover, we’re also able to
control whether the function is one for determining where the neuron activates or deactivates.

^^^^^ THE ABOVE ARE ALL WITH JUST ONE NEURON WE WILL EXPLORE MULTI-DIMENSIONAL ReLU ACTIVATIONS NEXT.
"""
# WITH MULTI-DIMENSIONAL neurons is were the magic happens.

"""
Now we see some fairly interesting behavior. The bias of the second neuron indeed shifted the
overall function, but, rather than shifting it horizontally, it shifted the function vertically. What
then might happen if we make that 2nd neuron’s weight -2 rather than 1?

What we have here is a neuron that has both an activation and a
deactivation point. When both neurons are activated, when their “area of effect” comes into play,
they produce values in the range of the granular, variable, and output. If any neuron in the pair is
inactive, the pair will produce non-variable output:

"""


# WILL USE A SNIPPET OF THIS CODE in the P.5

import numpy as np
import random
# WHen we import something as a name we are calling it sum
# Capital x is just a common practice dw
np.random.seed(0)

X = [[1, 2, 3, 2.5], [2., 5., -1., 2], [-1.5, 2.7, 3.3, -0.8]]
# The Self keyword just means that we can pass it in for a object.
#The __init__ BASICALLY IS JUST A CONSTRCUTOR KEYWORD.
# Look at __init__Self.py for


class LayerThick:
    def __init__(self, n_inputs, n_neurons):
        # Basically we are intializing out Inputs and Weights
        # We want the shape too. 
        # WHen a layer is created we need 2 things...
        # WHat is the size of the inputs and how many neurons we want in it.
        # >>>>>> n_inputs and N_neurons in the .randn function are the size of matric you wanna create
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        #we multiplieing the inputs and neurons by 0.1 because we want the values to be between 0 and 1
        self.biases = np.zeros((1, n_neurons), dtype=int)
        self.n_neurons = n_neurons

    def forward(self, inputs):
        # This function makes the NN move <"FORWARD">:
        # Calculates the outputs from the fucntuion above.
        self.output = np.dot(inputs, self.weights) + self.biases


# Size four because we have 4 elemnts inside each vector/matrix for our X<Inputs>
# The second number which are the neurons can be bascially nay number for now because nothing creates neurons yet.
layer1 = LayerThick(4, 5)
#this is our first layer but the parameters for our second layer are...
# The first parameters for the second layer(THE INPUTS) have to be the outputs of the fisrt layer.
layer2 = LayerThick(5, 2)

layer1.forward(X)
print(layer1.output)


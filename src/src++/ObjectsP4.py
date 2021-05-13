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
        self.weights = 0.1 * np.random.randn(n_inputs,n_neurons) # >>>>>> n_inputs and N_neurons in the .randn function are the size of matric you wanna create
        #we multiplieing the inputs and neurons by 0.1 because we want the values to be between 0 and 1
        self.biases = np.zeros((1,n_neurons), dtype = int)
        self.n_neurons = n_neurons
    def forward(self,inputs):
        # This function makes the NN move <"FORWARD">:
        # Calculates the outputs from the fucntuion above.
        self.output = np.dot(inputs, self.weights) + self.biases
        
# Size four because we have 4 elemnts inside each vector/matrix for our X<Inputs>
# The second number which are the neurons can be bascially nay number for now because nothing creates neurons yet.
layer1 = LayerThick(4,5)
#this is our first layer but the parameters for our second layer are...
# The first parameters for the second layer(THE INPUTS) have to be the outputs of the fisrt layer.
layer2 = LayerThick(5,2)

layer1.forward(X)
print(layer1.output)


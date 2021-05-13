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

# WHen we import something as a name we are calling it sum
# Capital x is just a common practice dw


# The Softmax activation function.
# We will use this for the output layer.
# The main reason we need another function for the output layer is we need osmehting to create a probability to that the Network can make a prediction.
# We can make a prediction maybe correctly whenusing the ReLU and when we ahve all positive values but when we have a negative value, Relu clips negatives to zero.
# SO both neurons are negative the network outputs are both zero. SO it cant be make a prediction.
# Hence we need a function whos equation doesnt clip the output to a zero.
# YAYYYY softmax.
#  \sigma(\vec{z})_{i}=\frac{e^{z_{i}}}{\sum_{j=1}^{K} e^{z_{j}}}
# We also expionetiate the function so we can mkae them non negative.
# Then we get a very high number, so then we "normalize" them (divide them) so we get a vector of small values.
# THEN WE ADD IT UP and get hte final number f that neuron.

"""
With a randomly-initialized model, or even a model initialized with more sophisticated
approaches, our goal is to train, or teach, a model over time. To train a model, we tweak the
weights and biases to improve the model’s accuracy and confidence. To do this, we calculate how
much error the model has. The loss function, also referred to as the cost function, is the
algorithm that quantifies how wrong a model is. Loss is the measure of this metric. Since loss is
the model’s error, we ideally want it to be 0.
"""
# Basically it talks to the Netowrk and it tells it how bad it did. Like a test score.
# The lower the loss funciton outputs the better our network did.

# Preferably we could clssify a (True, False) and then tell the network how correct he is (50% cux only true or false).
# We cant do that becasue we want the network to be confident in tis answer.
# As you can probably guess there are diffrebt types of Loss functions
# The softmax function outputs a probability distro so we need to Calculate loss with something that only comapres prob distros
# CATAGORIAL CROSS ENTROPY >>>>
# Why Categorical Cross Entropy >>> when introducing back propogation it becomes easier to change the values.
# Categories are like groups on types of data.
# One-Hot encoding>>>>>>
# Each output class(OT layer neuron) doesnt retrun a lable saying whatver you wnat to output, yet it will output an integer. 

import numpy as np
import nnfs
from nnfs.datasets import spiral_data
 # One Hot encdoing BASICALLY HELPS US SEE WHICH NEURON IS "HOT" or WHICH NEURON IS ACTIVATED.
# A vector is created for each category classes. All of the numbers are zeroes unless the coressponding class number's index in the vector is the number "1".
"""

softmax_output = [[0.7, 0.1, 0.2],
                  [0.1, 0.5, 0.4],
                  [0.02, 0.9, 0.08]]
"""
"""
The first value, 0, in class_targets means the first softmax output distribution’s intended
prediction was the one at the 0th index of [0.7, 0.1, 0.2]; the model has a 0.7 confidence
score that this observation is a dog. This continues throughout the batch, where the intended target
of the 2nd softmax distribution, [0.1, 0.5, 0.4], was at an index of 1; the model only has a
0.5 confidence score that this is a cat — the model is less certain about this observation. In the
last sample, it’s also the 2nd index from the softmax distribution, a value of 0.9 in this case — a
pretty high confidence
"""

inputs = [4.8, 1.21, 2.385]

nnfs.init()


def gen_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4,
                        points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y
# ================================================================ - main classes

class LayerThick:
    def __init__(self, n_inputs, n_neurons):
        # Basically we are intializing out Inputs and Weights
        # We want the shape too.
        # WHen a layer is created we need 2 things...
        # WHat is the size of the inputs and how many neurons we want in it.
        # >>>>>> n_inputs and N_neurons in the .randn function are the size of matric you wanna create
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        # we multiplieing the inputs and neurons by 0.1 because we want the values to be                            between 0 and 1
        self.biases = np.zeros((1, n_neurons), dtype=int)
        self.n_neurons = n_neurons

    def forward(self, inputs):
        # This function makes the NN move <"FORWARD">:
        # Calculates the outputs from the fucntuion above.
        self.output = np.dot(inputs, self.weights) + self.biases


class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        # This is the ReLU


class Softmax():
    def forward(self, inputs):
        #self.output = exp_values / np.sum(exp_values)
        # ALl of this would work with a just a 1D Vector yet it wont work with a Matrix/Batch because it will add every single number.
        # We need to specify what numbers columns to multiply...
        # We also need to divide it by the correct alinements so... 4
        # Plus the number can get too big and overflow.
        # We need to subtract the max value from the
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


# Common loss class
class Loss:
    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):
        # Calculate sample losses
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        # Return loss
        return data_loss
        # Cross-entropy loss

#Loss_CategoricalCrossentropy
class CCE(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

    # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )
        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 3 output values
layer1 = LayerThick(2, 3)
activation1 = ReLU()
# ^^^ created the first layer
layer2 = LayerThick(3, 3)
activation2 = Softmax()
# Creatig the second layer

costfunc = CCE()

# Perform a forward pass of our training data through this layer
layer1.forward(X)
# Perform a forward pass through activation function
# it takes the output of first dense layer here
activation1.forward(layer1.output)

# Perform a forward pass through second Dense layer
# it takes outputs of activation function of first layer as inputs
layer2.forward(activation1.output)
activation2.forward(layer2.output)

print(activation2.output[:5])
# WHen we created the smaple data class there are 300 data points.
loss = costfunc.calculate(activation2.output, y)
print("Loss:",loss)
"""
Again, we get ~0.33 values since the model is random, and its average loss is also not great for
these data, as we’ve not yet trained our model on how to correct its errors.
"""

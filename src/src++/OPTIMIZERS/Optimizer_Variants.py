import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data
from nnfs.datasets import vertical_data
import numpy as np
import math
import nnfs
nnfs.init()
e = math.e


# ================================================================ - main classes


class LayerThick:
    def __init__(self, n_inputs, n_neurons):
        # Basically we are intializing out Inputs and Weights
        # We want the shape too.
        # WHen a layer is created we need 2 things...
        # WHat is the size of the inputs and how many neurons we want in it.
        # >>>>>> n_inputs and N_neurons in the .randn function are the size of matric you wanna create
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        # we multiplieing the inputs and neurons by 0.1 because we want the values to be between 0 and 1
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        # This function makes the NN move <"FORWARD">:
        # Calculates the outputs from the fucntuion above.
        self.output = np.dot(inputs, self.weights) + self.biases
    # BACK PROP METHOD for well prurposed gradient descent

    def backward(self, dvalues):
        # Gradientiantial Parameters
        # >>> .T ====== IS TO KEEP IT TRANSPOSED
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        self.inputs = inputs
        # This is the ReLU
    # Backward pass

    def backward(self, dvalues):
        # Since we need to modify the original variable,
        # let's make a copy of the values first
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0


class Softmax():
    def forward(self, inputs):
        #self.output = exp_values / np.sum(exp_values)
        # ALl of this would work with a just a 1D Vector yet it wont work with a Matrix/Batch because it will add every single number.
        # We need to specify what numbers columns to multiply...
        # We also need to divide it by the correct alinements so... 4
        # Plus the number can get too big and overflow.
        # We need to subtract the max value from the
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output and
            jacobian_matrix = np.diagflat(single_output) - \
                np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,
                                         single_dvalues)

# Common loss class


class Loss:
    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, yn):
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

    def backward(self, dvalues, y_true):
       # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples
# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step


class Softmax_CCE():

    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Softmax()
        self.loss = CCE()

    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)

    # Backward pass

    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# THIS is one optimizer thats all =. I will be learning about 4 diffrent optimizers. But i will only use one.


class SGD:
    # Make the optimizer
    # Intialize the learning rate which is 1.0 for now.
    """
    Current learning rate, and self.learning_rate is now the initial learning
    rate. We also added attributes to track the decay rate and the number of iterations that the
    optimizer has gone through.
    """
    # When we write said program to make this we end up with Decay ----
    # Decay is a way the learning rate can go down by itself
    # Momentum is when we take the average if the last few steps taken and add that to the push the function outside of the local minuma.

    # The learning rate is a Hyperparameter that shows how fast your netowrk can adapt to the the situation data.
    # The lower it is the better. Its like a speed calculation.

    # Decay Rate,which steadily decays the learning rate per batch or epoch
    def __init__(self, learning_rate=1., decay=0, momentum=0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        # momentum is when we avergae all the gradeint f th elast few steps and give the GD model a little push to get over this local minumum.
     # Call once before any parameter updates
    """
    This method, if we have a decay rate other than 0, will update our self.current_learning_rate
    using the prior formula
    """

    def learning_update(self):
        if not self.decay == 0:
            self.current_learning_rate = self.learning_rate * \
                (1 / (1 + self.decay * self.iterations))

    # Update the Parameters of the network.
    def update(self, layer):
        if not self.momentum == 0:
            # If layer does not contain momentum arrays, create them
            # filled with zeros
            # Hassatr is a boolean vlaue that will check if the weight_montums is in the object layer.
            if hasattr(layer, 'weight_momentums') == False:
                layer.weight_momentums = np.zeros_like(layer.weights)
                # If there is no momentum array for weights
                # The array doesn't exist for biases yet either.
                layer.bias_momentums = np.zeros_like(layer.biases)

            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with
            # current gradients
            weight_updates = self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            # Build bias updates
            bias_updates = self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        # Vanilla SGD updates (as before momentum update)
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        # Update weights and biases using either
        # vanilla or momentum updates
        layer.weights += weight_updates
        layer.biases += bias_updates

    # This Method will add to our self.iterations tracking

    def iteration_update(self):
        self.iterations += 1

# Adagrad optimizer



#+=============================================#
#Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 64 output values
layer1 = LayerThick(2, 64)

# Create ReLU activation (to be used with Dense layer):
activation1 = ReLU()

# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values (output values)
layer2 = LayerThick(64, 3)

# Create Softmax classifier's combined loss and activation
costfunc = Softmax_CCE()

# Create Optimizer
optimizer = SGD(decay=1e-3, momentum=0.9)
# FOr the optmizer we will try to reuse the iterative optimizations strat but we will do it better.
#========================================
for epoch in range(10001):
    # Perform a forward pass of our training data through this layer
    layer1.forward(X)

    # Perform a forward pass through activation function
    # takes the output of first dense layer here
    activation1.forward(layer1.output)

    # Perform a forward pass through second Dense layer
    # takes outputs of activation function of first layer as inputs
    layer2.forward(activation1.output)

    # Perform a forward pass through the activation/loss function
    # takes the output of second dense layer here and returns loss
    loss = costfunc.forward(layer2.output, y)

    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(costfunc.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    # Print accuracy
    if epoch % 100 == 0:
        print(f'epoch/iteration: {epoch}, ' +
              f'accuracy: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}, '
              f'lr: {optimizer.current_learning_rate}')
    # Backward pass
    costfunc.backward(costfunc.output, y)
    layer2.backward(costfunc.dinputs)
    activation1.backward(layer2.dinputs)
    layer1.backward(activation1.dinputs)

    # UPdateee
    optimizer.learning_update()
    optimizer.update(layer1)
    optimizer.update(layer2)
    optimizer.iteration_update()

    #^^^^^  Each full pass through all of the training data is called an epoch


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
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        # Basically we are intializing out Inputs and Weights
        # We want the shape too.
        # WHen a layer is created we need 2 things...
        # WHat is the size of the inputs and how many neurons we want in it.
        # >>>>>> n_inputs and N_neurons in the .randn function are the size of matric you wanna create
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        # we multiplieing the inputs and neurons by 0.1 because we want the values to be between 0 and 1
        self.biases = np.zeros((1, n_neurons))
        # Setting the weights/biases equal to the constructor params.
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs):
        self.inputs = inputs
        # This function makes the NN move <"FORWARD">:
        # Calculates the outputs from the fucntuion above.
        self.output = np.dot(inputs, self.weights) + self.biases
    # BACK PROP METHOD for well prurposed gradient descent

    def backward(self, dvalues):
        # Gradientiantial Parameters
        # >>> .T ====== IS TO KEEP IT TRANSPOSED
        # Th axis is just a sub set specification parameter we shoose to include when adding values in certain matrices
        # An axis of 1 means it will add each vector in a matrix
        # Axis = none means it will add it fully
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * \
                self.weights
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * \
                self.biases

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


class Dropout:
    def __init__(self, rate):
        self.rate = 1-rate
        # Here we are creating a succesion rate 1- (0.1)== 0.9

    def forward(self, inputs):
        # Save input values
        self.inputs = inputs
        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate,
                                              size=inputs.shape) / self.rate
        # Apply mask to output values
        self.output = inputs * self.binary_mask

    # Backward pass
    def backward(self, dvalues):
        # Gradient on values
        self.dinputs = dvalues * self.binary_mask


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
            jacobian_matrix = np.diagflat(
                single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,
                                         single_dvalues)

# Common loss class


class Loss:
    # Calculates the data and regularization losses
    # given model output and ground truth values
    def reg_loss(self, layer):
        # This value is like a penalty so it provides
        # 0 is the default
        reg_loss = 0

        if layer.weight_regularizer_l1 > 0:
            reg_loss += layer.weight_regularizer_11 * \
                np.sum(np.abs(layer.weights))

        if layer.weight_regularizer_l2 > 0:
            reg_loss += layer.weight_regularizer_l2 * \
                np.sum(layer.weights * layer.weights)

        if layer.bias_regularizer_l2 > 0:
            reg_loss += layer.bias_regularizer_l1 * \
                np.sum(np.abs(layer.biases))

        if layer.bias_regularizer_l2 > 0:
            reg_loss += layer.bias_regularizer_l2 * \
                np.sum(layer.biases * layer.biases)

        return reg_loss

    def calculate(self, output, yn):
        # Calculate sample losses
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        # Return loss
        return data_loss
        # Cross-entropy loss


class CCE(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # We do this confidently becuase the division will not occur if it is zero.

    # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) <= 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) >= 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
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

    #Current learning rate, and self.learning_rate is now the initial learning
    #rate. We also added attributes to track the decay rate and the number of iterations that the
    #optimizer has gone through.

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
        # vanilla    or momentum updates
        layer.weights += weight_updates
        layer.biases += bias_updates

    # This Method will add to our self.iterations tracking

    def iteration_update(self):
        self.iterations += 1


class Adam:
    # The main differnece between the SGD and the Adam Optimizer is that...
    #

    # Initialize optimizer - set settings
    # Make the optimizer
    # Intialize the learning rate which is 1.0 for now.

    #Current learning rate, and self.learning_rate is now the initial learning
    #rate. We also added attributes to track the decay rate and the number of iterations that the
    #optimizer has gone through.

    # When we write said program to make this we end up with Decay ----
    # Decay is a way the learning rate can go down by itself
    # Momentum is when we take the average if the last few steps taken and add that to the push the function outside of the local minuma.

    # The learning rate is a Hyperparameter that shows how fast your netowrk can adapt to the the situation data.
    # The lower it is the better. Its like a speed calculation.

    # Decay Rate,which steadily decays the learning rate per batch or epoch

    # Initialize optimizer - set settings
    # The Cache is a history for the
    def __init__(self, learning_rate=0.001, decay=0.9, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        # To prevent the value to be divided by zero and get too low.
        self.beta_1 = beta_1
        # To correct the cache and provide a correction method
        self.beta_2 = beta_2
        # To correct the momentum and provide a momentum updater.
    # Call once before any parameter updates
    # IF we any other decay rate than zero then update the learning rate.

    def learning_update(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update(self, layer):

        # If layer does not contain cache arrays,
        # create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum  with current gradients
        layer.weight_momentums = self.beta_1 * \
            layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * \
            layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * layer.dweights**2

        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * layer.dbiases**2
        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / \
            (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / \
            (np.sqrt(bias_cache_corrected) + self.epsilon)

    # Call once after any parameter updates
    def iteration_update(self):
        self.iterations += 1


#+=============================================#
#Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create Dense layer with 2 input features and 64 output values
layer1 = LayerThick(2, 64, weight_regularizer_l2=5e-4,
                    bias_regularizer_l2=5e-4)

# Create ReLU activation (to be used with Dense layer):
activation1 = ReLU()


dropout1 = Dropout(0.1)
# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values (output values)
layer2 = LayerThick(64, 3)

# Create Softmax classifier's combined loss and activation
costfunc = Softmax_CCE()

# Create Optimizer
optimizer = Adam(learning_rate=0.02, decay=1e-5)
# FOr the optmizer we will try to reuse the iterative optimizations strat but we will do it better.
#========================================
for epoch in range(10001):
    # Perform a forward pass of our training data through this layer
    layer1.forward(X)

    # Perform a forward pass through activation function
    # takes the output of first dense layer here
    activation1.forward(layer1.output)

    dropout1.forward(activation1.output)
    # Perform a forward pass through second Dense layer
    # takes outputs of activation function of first layer as inputs
    layer2.forward(dropout1.output)

    # Perform a forward pass through the activation/loss function
    # takes the output of second dense layer here and returns loss
    pen_loss = costfunc.forward(layer2.output, y)
    # Calculate the penalty

    reg_loss = costfunc.loss.reg_loss(layer1) + costfunc.loss.reg_loss(layer2)

    loss = pen_loss + reg_loss

    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(costfunc.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = (np.mean(predictions == y))*100

    # Print accuracy
    if epoch % 100 == 0:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f} (' +
              f'data_loss: {pen_loss:.3f}, ' +
              f'reg_loss: {reg_loss:.3f}), ' +
              f'lr: {optimizer.current_learning_rate}')
    # Backward pass
    costfunc.backward(costfunc.output, y)
    layer2.backward(costfunc.dinputs)
    dropout1.backward(layer2.dinputs)
    activation1.backward(dropout1.dinputs)

    layer1.backward(activation1.dinputs)

    # UPdateee
    optimizer.learning_update()
    optimizer.update(layer1)
    optimizer.update(layer2)
    optimizer.iteration_update()


# Validate the model

# Create test dataset
X_test, y_test = spiral_data(samples=100, classes=3)

# Perform a forward pass of our testing data through this layer
layer1.forward(X_test)

# Perform a forward pass through activation function
# takes the output of first dense layer here
activation1.forward(layer1.output)

# Perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs
layer2.forward(activation1.output)

# Perform a forward pass through the activation/loss function
# takes the output of second dense layer here and returns loss
loss = costfunc.forward(layer2.output, y_test)

# Calculate accuracy from output of activation2 and targets
# calculate values along first axis
predictions = np.argmax(costfunc.output, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions == y_test)

print(f'validation, acc: {accuracy:.3f}, Validation loss: {loss:.3f}')

#^^^^^  Each full pass through all of the training data is called an epoch

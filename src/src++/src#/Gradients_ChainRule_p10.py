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

# Introducing Optimization
# Optimization is a way to adjust the weights and biases to get a better result and to get higher accuracy abnd lower cost function value

"""
Randomly changing and searching for optimal weights and biases did not prove fruitful for one
main reason: the number of possible combinations of weights and biases is infinite, and we need
something smarter than pure luck to achieve any success. Each weight and bias may also have
different degrees of influence on the loss — this influence depends on the parameters themselves
as well as on the current sample, which is an input to the first layer. These input values are then
multiplied by the weights, so the input data affects the neuron’s output and affects the impact that
the weights make on the loss. The same principle applies to the biases and parameters in the next
layers, taking the previous layer’s outputs as inputs. This means that the impact on the output
values depends on the parameters as well as the samples — which is why we are calculating the
loss value per each sample separately. Finally, the function of how a weight or bias impacts the
overall loss is not necessarily linear. In order to know how to adjust weights and biases, we first
need to understand their impact on the loss.
"""
"""
The derivatives that we’ve solved so far have been
cases where there is only one independent variable in the function — that is, the result depended
solely on, in our case, x. However, our neural network consists, for example, of neurons, which
have multiple inputs. Each input gets multiplied by the corresponding weight (a function of 2
parameters), and they get summed with the bias (a function of as many parameters as there are
inputs, plus one for a bias). As we’ll explain soon in detail, to learn the impact of all of the inputs,
weights, and biases to the neuron output and at the end of the loss function, we need to calculate
the derivative of each operation performed during the forward pass in the neuron and the whole
model. To do that and get answers, we’ll need to use the chain rule.
"""
"""
Derivatives of a function f(x):
    - how fast f changes around x (slope)
    - will f increase or decrease if we increse x
    - https://www.youtube.com/watch?v=3p09AyD-eWI


"""

# The Partial derivative is a calculation of how much impact a neuron has on a function output.
# Basically like the rate of change.

# Gradients are just a 1d array / Vector of the partial derivatives of an input.

# Chain rule: The forward pass through our model is a chain of functions similar to these examples. We are passing in samples, the data flows through all of the layers, and activation functions to form a CHAIN.

# Chain Rule is kind of like the transitive property.
# Suppose we are a collecting 

# Forward pass
x = [1.0, -2.0, 3.0]  # input values
w = [-3.0, -1.0, 2.0]  # weights
b = 1.0  # bias
# Multiplying inputs by weights
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]
print(xw0, xw1, xw2, b)
# Adding weighted inputs and a bias
z = xw0 + xw1 + xw2 + b
print(z)
# ReLU activation function
y = max(z, 0)
print(y)

"""
This is the full forward pass through a single neuron and a ReLU activation function. Let’s treat
all of these chained functions as one big function which takes input values (x), weights (w), and
bias (b), as inputs, and outputs y. This big function consists of multiple simpler functions — there
is a multiplication of input values and weights, sum of these values and bias, as well as a max
function as the ReLU activation — 3 chained functions in total:

"""


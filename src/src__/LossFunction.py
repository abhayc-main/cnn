
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
# Since both neurons are negative the network outputs are both zero. SO it cant be make a prediction.
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

# Preferably we could clssify a (True, False) and then tell the network how correct he is (50% cux only true or false).
# We cant do that becasue we want the network to be confident in tis answer.
# As you can probably guess there are diffrebt types of Loss functions
# The softmax function outputs a probability distro so we need to Calculate loss with something that only comapres prob distros
# CATAGORIAL CROSS ENTROPY >>>>





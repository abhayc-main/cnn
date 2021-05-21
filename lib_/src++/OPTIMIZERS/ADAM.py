# BEST OF THEM ALL
"""
Adam, short for Adaptive Momentum, is currently the most widely-used optimizer and is built
atop RMSProp, with the momentum concept from SGD added back in. This means that, instead
of applying current gradients, we’re going to apply momentums like in the SGD optimizer with
momentum, then apply a per-weight adaptive learning rate with the cache as done in RMSProp.

Adam combines the best of RMS prop and SGD with momentum.


"""

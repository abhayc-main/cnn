import matplotlib.pyplot as plt
import numpy as np
def f(x):
    return 2*x**2
# np.arange(start, stop, step) to give us smoother line
x = np.arange(0, 5, 0.001)
y = f(x)
plt.plot(x, y)
plt.show()

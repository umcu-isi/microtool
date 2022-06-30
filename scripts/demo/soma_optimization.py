from microtool.optimize import SOMA
import numpy as np
from scipy.optimize import minimize


def ackley(x):
    n = float(len(x))
    ninv = 1 / n
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return 20 + np.exp(1) - (20 * np.exp(-.2 * np.sqrt(ninv * sum1))) - np.exp(ninv * sum2)


# # Setting the control parameters
soma_optimizer = SOMA()
print(soma_optimizer)

# setting optimization parameters
Ndim = 2
x0 = np.zeros(Ndim)
domain = (-500, 500)
bounds = np.array([domain for i in range(Ndim)])

for i in range(10):
    result = minimize(ackley, x0, method=soma_optimizer, bounds=bounds)
    print(result)

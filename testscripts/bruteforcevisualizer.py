import numpy as np
from numba import jit
from scipy.optimize import minimize

from microtool.optimize import BruteForce


@jit
def ackley(x):
    n = float(len(x))
    ninv = 1 / n
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return 20 + np.exp(1) - (20 * np.exp(-.2 * np.sqrt(ninv * sum1))) - np.exp(ninv * sum2)


# setting optimization parameters
Ndim = 10
x0 = np.zeros(Ndim)
domain = (-.5, .5)
bounds = [domain for i in range(Ndim)]
# setting the optimizer
brute_force = BruteForce(Ns=10)
result = minimize(ackley, x0, method=brute_force, bounds=bounds)
print(result)

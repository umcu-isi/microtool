from matplotlib import pyplot as plt
from microtool.optimize import BruteForce
from scipy.optimize import minimize
import numpy as np
from numba import jit


@jit
def ackley(x):
    n = float(len(x))
    ninv = 1 / n
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return 20 + np.exp(1) - (20 * np.exp(-.2 * np.sqrt(ninv * sum1))) - np.exp(ninv * sum2)


# setting optimization parameters
Ndim = 2
x0 = np.zeros(Ndim)
domain = (-50, 50)
bounds = [domain for i in range(Ndim)]
# setting the optimizer
brute_force = BruteForce(Ns=1000)
result = minimize(ackley, x0, method=brute_force, bounds=bounds)

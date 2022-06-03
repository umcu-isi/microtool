from microtool.optimize import brute_force
from scipy.optimize import minimize
import numpy as np


def ackley(x):
    n = float(len(x))
    ninv = 1 / n
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return 20 + np.exp(1) - (20 * np.exp(-.2 * np.sqrt(ninv * sum1))) - np.exp(ninv * sum2)


# setting optimization parameters
Ndim = 2
x0 = np.zeros(Ndim)
domain = (-500, 500)
bounds = [domain for i in range(Ndim)]

minimize(ackley, x0, method=brute_force, bounds=bounds ,options={"show_plot": True, "Ns":100})

import numpy as np
from scipy.optimize import Bounds, minimize

from microtool.optimize.methods import SOMA


def ackley(x):
    """
    The ackley function is a well known test problem in optimization. It has its minimum at x = (0,..,0)
    """
    n = float(len(x))
    ninv = 1 / n
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return 20 + np.exp(1) - (20 * np.exp(-.2 * np.sqrt(ninv * sum1))) - np.exp(ninv * sum2)


def test_ackley_problem():
    method = SOMA()
    # setting optimization parameters
    Ndim = 2
    x0 = np.ones(Ndim)
    domain = (-5, 5)
    bounds_array = np.array([domain for i in range(Ndim)])
    bounds = Bounds(bounds_array[:, 0], bounds_array[:, 1])

    result = minimize(ackley, x0, method=method, bounds=bounds)

    solution = np.zeros(Ndim)

    np.testing.assert_allclose(solution, result.x, atol=1e-6)


def test_multi_constraint():
    pass

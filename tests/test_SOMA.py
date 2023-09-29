import numpy as np
from scipy.optimize import Bounds, minimize, NonlinearConstraint

from microtool.optimize.methods import SOMA


def test_ackley_problem():
    def ackley(x):
        """
        The ackley function is a well known test problem in optimization. It has its minimum at x = (0,..,0)
        """
        n = float(len(x))
        ninv = 1 / n
        sum1 = np.sum(x ** 2)
        sum2 = np.sum(np.cos(2 * np.pi * x))
        return 20 + np.exp(1) - (20 * np.exp(-.2 * np.sqrt(ninv * sum1))) - np.exp(ninv * sum2)

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
    def rosenbrock(x):
        """ The two dimensional rosenbrock function. The minimum is located at (1,1)"""
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    # Constraints
    def constraint1(x):
        return x[0] + x[1] - 1.5  # x + y >= 1.5

    def constraint2(x):
        return 2.5 - x[0] - x[1]  # x + y <= 2.5

    def constraint3(x):
        return x[0] - x[1] + 0.5  # x - y >= -0.5

    def constraint4(x):
        return 0.5 - x[0] + x[1]  # x - y <= 0.5

    # Defining the Nonlinear Constraints
    nlc1 = NonlinearConstraint(constraint1, 0, np.inf)
    nlc2 = NonlinearConstraint(constraint2, 0, np.inf)
    nlc3 = NonlinearConstraint(constraint3, 0, np.inf)
    nlc4 = NonlinearConstraint(constraint4, 0, np.inf)

    domain = (-5., 5.)
    bounds_array = np.array([domain] * 2)
    bounds = Bounds(bounds_array[:, 0], bounds_array[:, 1])
    x0 = [0., 0.]
    result = minimize(rosenbrock, x0, method=SOMA(), bounds=bounds, constraints=[nlc1, nlc2, nlc3, nlc4])

    solution = [1., 1.]
    np.testing.assert_allclose(solution, result.x, atol=1e-1)

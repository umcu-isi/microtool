from typing import Union, List, Sequence

import numpy as np
from numba import njit

Number = Union[np.ndarray, float]


def is_smaller_than_with_tolerance(number: Number, lower_bound: Number, tolerance=1e-17) -> Union[bool, np.ndarray]:
    """
    This function returns True if number is strictly smaller than a lower bound where we include a tolerance for which
    we consider number to be equal to lower bound and return False.

    :param number: The number which we wish to check
    :param lower_bound: The lower bound
    :param tolerance: The tolerance that we allow for equality with the lower bound
    :return: True if strictly lower than lower bound with a given tolerance. False otherwise
    """
    # array with values close to lb true
    close_to_lb = np.isclose(number, lower_bound, atol=tolerance)

    # normal comparison array
    less_than_lb = number < lower_bound

    # set values within tolerance to False (we consider this equality with the bound)
    less_than_lb[close_to_lb] = False
    return less_than_lb


def is_higher_than_with_tolerance(number: Number, upper_bound: Number, tolerance=1e-17) -> Union[bool, np.ndarray]:
    """
    This function returns True if number is strictly larger than a lower bound where we include a tolerance for which
    we consider number to be equal to upperbound and return False.

    :param number: The number which we wish to check
    :param upper_bound: The upper bound
    :param tolerance: The tolerance that we allow for equality with the lower bound
    :return: True if strictly higher than higher with a given tolerance. False otherwise
    """
    # Using math we reuse the function above
    return is_smaller_than_with_tolerance(-1. * number, -1. * upper_bound, tolerance)


# TODO: is this function necessary? Is its use a bottleneck and is this significantly faster than np.cross?
@njit
def cartesian_product(jac: np.ndarray):
    # number of parameters (we use N for tissue parameters and M for Measurements)
    m, n = jac.shape
    derivative_term = np.zeros((n, n, m))
    for i in range(n):
        for j in range(n):
            derivative_term[i, j, :] = jac[:, i] * jac[:, j]
    return derivative_term


# TODO: is this function necessary? Is its use a bottleneck and is this significantly faster than np.diagonal?
@njit
def diagonal(square: np.ndarray) -> np.ndarray:
    size, _ = square.shape
    out = np.zeros(size)
    for i in range(size):
        out[i] = square[i, i]

    return out


def largest_real_cbrt(a2: Union[float, np.ndarray], a1: Union[float, np.ndarray], a0: Union[float, np.ndarray]) -> \
        Union[float, np.ndarray]:
    """
    Returns the largest real root of the cubic equation x³ + a2 * x² + a1 * x + a0 = 0.

    See https://mathworld.wolfram.com/CubicFormula.html for details on solving the cubic analytically.

    :param a2: Coefficient(s) of the quadratic term
    :param a1: Coefficient(s) of the linear term
    :param a0: Constant(s)
    :return: The largest real root(s) or NaN if there is no solution.
    """
    q = (3 * a1 - a2 ** 2) / 9
    r = (9 * a2 * a1 - 27 * a0 - 2 * a2 ** 3) / 54
    sqrt_d = np.sqrt(q ** 3 + r ** 2 + 0j)  # adding 0j to allow for computation with complex numbers
    s = (r + sqrt_d) ** (1 / 3)
    t = (r - sqrt_d) ** (1 / 3)

    a = -(1 / 3) * a2
    b = -0.5 * (s + t)
    c = 0.5j * np.sqrt(3) * (s - t)

    z1 = a + (s + t)
    z2 = a + b + c
    z3 = a + b - c

    roots = np.stack([z1, z2, z3])

    # we consider numbers real if imaginary part is zero to some tolerance.
    real_roots = np.isclose(roots.imag, np.zeros(roots.shape))
    roots[~real_roots] = np.nan
    largest_real_roots = np.nanmax(roots.real, axis=0)

    return largest_real_roots


def real_cbrt(a2: Union[float, np.ndarray], a1: Union[float, np.ndarray], a0: Union[float, np.ndarray]) -> \
        np.ndarray:
    """
    Returns the real root of the cubic equation x³ + a2 * x² + a1 * x + a0 = 0, assuming that there is only one real
     root.

    See https://mathworld.wolfram.com/CubicFormula.html for details on solving the cubic analytically.

    :param a2: Coefficient(s) of the quadratic term
    :param a1: Coefficient(s) of the linear term
    :param a0: Constant(s)
    :return: The real root or NaN if there is no solution.
    """
    q = (3 * a1 - a2 ** 2) / 9
    r = (9 * a2 * a1 - 27 * a0 - 2 * a2 ** 3) / 54
    sqrt_d = np.sqrt(q ** 3 + r ** 2 + 0j)  # adding 0j to allow for computation with complex numbers
    s = (r + sqrt_d) ** (1 / 3)
    t = (r - sqrt_d) ** (1 / 3)

    root = -(1 / 3) * a2 + s + t

    # We consider numbers real if imaginary part is zero to some tolerance.
    real = np.isclose(root.imag, np.zeros_like(root))
    root = np.where(real, root.real, np.nan)

    return root


def eval_polynomial(c: List[Union[float, np.ndarray]], x: Union[float, np.ndarray]):
    """
    Evaluates the polynomial y = c[0] + c[1] x + c[2] x² + ...

     :param c: List of coefficients or coefficient vectors.
     :param x: x
     :return: y
    """
    y = None
    for i, coefficients in enumerate(c):
        if coefficients is not None:
            if y is None:
                y = coefficients * x**i
            else:
                y += coefficients * x**i

    return y


def newton_polynomial_root(c: List[Union[float, np.ndarray]],
                           x0: Union[float, np.ndarray],
                           n: int = 8) -> np.ndarray:
    """
    Uses Newton's method to find a root of a polynomial with coefficients c, starting at x = x0:
     0 = c[0] + c[1] x + c[2] x² + ...

     :param c: List of coefficients or coefficient vectors. Use None instead of 0 to exclude a term.
     :param x0: Initial guess.
     :param n: Number of iterations.
     :return: Result of Newton's method after n iterations.
    """
    # Calculate the coefficients for the derivative of the polynomial.
    d = [None if x is None else i * x for i, x in enumerate(c[1:], 1)]

    x = x0 * np.array([1.])  # Make sure x is an array (without casting, which would make the tests fail).
    for _ in range(n):
        y = eval_polynomial(c, x)
        dy = eval_polynomial(d, x)
        nonzero = dy != 0
        x[nonzero] -= y[nonzero] / dy[nonzero]

    return x


def random_uniform(lower_bound: float = 0, upper_bound: float = 1, size: Union[int, Sequence] = 1) -> np.ndarray:
    """
    A pint-compatible version of np.random.uniform. Allows us to generate random numbers with units during testing.

     :param lower_bound: Lower bound of the uniform distribution.
     :param upper_bound: Upper bound of the uniform distribution.
     :param size: Size of each dimension.
     :return: Array of random numbers.
    """
    shape = (size,) if isinstance(size, int) else size
    return lower_bound + np.random.rand(*shape) * (upper_bound - lower_bound)

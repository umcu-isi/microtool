"""
The following expressions were found analytically


"""

import numpy as np

# 1/ ms mT
gamma = 42.57747892 * 2 * np.pi
# ms?
t90 = 4
t180 = 6
# s /micro m^2 ?
b = 0.05
# ms?
t_half = 14

G_max = 200e-6  # mt/um
S_max = 1300e-6  # mt/um/ms
t_rise = G_max / S_max


#

def main():
    print(minimal_echo_time(delta_max()))


def minimal_echo_time(delta_max):
    delta = np.linspace(1e-6, delta_max, num=1000)
    return np.min(echo_time(delta))


def echo_time(delta):
    return 0.5 * t90 + Delta(delta) + delta + t_rise + t_half


def Delta(delta):
    B = b / (gamma ** 2 * G_max ** 2)
    return (B - t_rise / 30) * delta ** (-2) + (t_rise / 6) * delta ** (-1) + delta / 3


def delta_max():
    return min(compute_delta_max_1(), compute_delta_max_2())


def compute_delta_max_2():
    """

    :param vars:
    :return:
    """
    T1 = 0.5 * t90 + 0.5 * t180 + t_rise
    T2 = 0.5 * t90 + t_rise + t_half

    B = b / (gamma ** 2 * G_max ** 2)

    a2 = -3 * (0.5 * T2 - T1)
    a1 = -0.25 * t_rise
    a0 = -1.5 * (B - t_rise / 30)
    return largest_real_cbrt(a2, a1, a0)


def compute_delta_max_1():
    """
    For computing delta max 1 we are considering the following equality

    TE == 0.5*t90 + delta + t_rise + t_half
    where TE = 0.5*t90 + Delta + delta + t_rise + t_half. (solving in delta)

    :return: delta_max 1
    """
    # The coefficients of the cubic equations a_i belongs to the term of delta^i
    p = t180 - 0.5 * t90 + t_rise + t_half
    a2 = (3 / 2) * p
    a1 = -t_rise / 4
    a0 = (-3 / 2) * (b / (gamma ** 2 * G_max ** 2)) + t_rise / 20
    return largest_real_cbrt(a2, a1, a0)
    # compute polynomial coefficients based on scan parameters


def largest_real_cbrt(a2: float, a1: float, a0: float) -> float:
    """
    Returns the largest real root of the cubic equation z^3 + a_2 * z^2 + a_1 * z + a_0 = 0.
    
    See https://mathworld.wolfram.com/CubicFormula.html for details on solving the cubic analytically.

    :param a2: Coefficient of the quadratic term
    :param a1: Coefficient of the linear term
    :param a0: Constant offset
    :return: The largest real root
    """
    Q = (3 * a1 - a2 ** 2) / 9
    R = (9 * a2 * a1 - 27 * a0 - 2 * a2 ** 3) / 54
    # adding 0j to allow for computation with complex numbers
    D = Q ** 3 + R ** 2 + 0j
    S = (R + np.sqrt(D)) ** (1. / 3.)
    T = (R - np.sqrt(D)) ** (1. / 3.)

    z1 = -(1 / 3) * a2 + (S + T)
    z2 = -(1 / 3) * a2 - 0.5 * (S + T) + 0.5j * np.sqrt(3) * (S - T)
    z3 = -(1 / 3) * a2 - 0.5 * (S + T) - 0.5j * np.sqrt(3) * (S - T)

    roots = np.array([z1, z2, z3])
    # we consider numbers real if imaginary part is zero to some tolerance
    real_root_mask = np.isclose(np.imag(roots), np.zeros(roots.shape))
    return np.real(max(roots[real_root_mask]))


if __name__ == '__main__':
    main()

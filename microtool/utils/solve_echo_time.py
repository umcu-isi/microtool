"""
All the times in this script are computed in ms. so we scale them at the final function.
"""
from copy import copy
from typing import Union

import numpy as np

from microtool.constants import GAMMA
from microtool.scanner_parameters import ScannerParameters

gamma_different_unit = 1e-3 * GAMMA


def minimal_echo_time(b, scanner_parameters: ScannerParameters):
    """

    :param b: The bvalue in seconds/millimeters^2
    :param scanner_parameters: Scan parameters
    :return: The minimal echo time in seconds
    """
    # copying so we dont actually change bvalues that this function takes
    b = copy(b)
    # for the zero b-values we just take the b = 50 s/mm^2 since it will be a suitable constraint
    b[b == 0] = 50
    # converting units
    b *= 1e3
    scanner_parameters = copy(scanner_parameters)
    scan_parameter_to_ms(scanner_parameters)

    delta_max = compute_delta_max(b, scanner_parameters)
    domain = np.linspace(1e-6, delta_max, num=1000)
    TE_min = np.min(echo_time(domain, b, scanner_parameters), axis=0)
    return TE_min * 1e-3


def scan_parameter_to_ms(scanner_parameters: ScannerParameters):
    # time parameters
    scanner_parameters.t_90 *= 1e3
    scanner_parameters.t_half *= 1e3
    scanner_parameters.t_180 *= 1e3
    # inverse time parameter
    scanner_parameters.S_max *= 1e-3


def echo_time(delta, b, scanner_parameters: ScannerParameters):
    # extracting the scan parameters
    G_max = scanner_parameters.G_max
    t_rise = scanner_parameters.t_rise
    t90 = scanner_parameters.t_90
    t_half = scanner_parameters.t_half
    # surrogate parameter for readability
    B = b / (gamma_different_unit ** 2 * G_max ** 2)
    Delta = (B - t_rise / 30) * delta ** (-2) + (t_rise / 6) * delta ** (-1) + delta / 3
    return 0.5 * t90 + Delta + delta + t_rise + t_half


def compute_delta_max(b, scanner_parameters: ScannerParameters):
    t180 = scanner_parameters.t_180
    t90 = scanner_parameters.t_90
    t_rise = scanner_parameters.t_rise
    t_half = scanner_parameters.t_half
    G_max = scanner_parameters.G_max

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
        a0 = (-3 / 2) * (b / (gamma_different_unit ** 2 * G_max ** 2)) + t_rise / 20
        return largest_real_cbrt(a2, a1, a0)

    def compute_delta_max_2():
        """
        0.5 TE == 0.5 t90 + 0.5 t180 + delta + trise

        :return: delta_max 2
        """
        T1 = 0.5 * t90 + 0.5 * t180 + t_rise
        T2 = 0.5 * t90 + t_rise + t_half

        B = b / (gamma_different_unit ** 2 * G_max ** 2)

        a2 = -3 * (0.5 * T2 - T1)
        a1 = -0.25 * t_rise
        a0 = -1.5 * (B - t_rise / 30)
        return largest_real_cbrt(a2, a1, a0)

    delta_max_1 = compute_delta_max_1()
    delta_max_2 = compute_delta_max_2()
    delta_max = np.min(np.array([delta_max_1, delta_max_2]).T, axis=1)
    return delta_max


def largest_real_cbrt(a2: Union[float, np.ndarray], a1: Union[float, np.ndarray], a0: Union[float, np.ndarray]) -> \
        Union[float, np.ndarray]:
    """
    Returns the largest real root of the cubic equation z^3 + a_2 * z^2 + a_1 * z + a_0 = 0.
    
    See https://mathworld.wolfram.com/CubicFormula.html for details on solving the cubic analytically.

    :param a2: Coefficient of the quadratic term
    :param a1: Coefficient of the linear term
    :param a0: Constant offset
    :return: The largest real root(s) in shape (N_parameter values, 1)
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

    # If there is a parameter array solutions are stored in shape (number of parametervalues, 3 or number of solutions)
    roots = np.array([z1, z2, z3]).T
    # we consider numbers real if imaginary part is zero to some tolerance
    real_root_mask = np.isclose(np.imag(roots), np.zeros(roots.shape))
    # Masking out the imaginary roots
    masked_roots = np.ma.masked_where(np.logical_not(real_root_mask), roots)
    largest_real_roots = np.max(np.real(masked_roots), axis=1)

    return largest_real_roots

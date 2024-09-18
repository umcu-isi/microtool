from copy import copy

import numpy as np

from .math import largest_real_cbrt
from .unit_registry import unit
from ..constants import GAMMA
from ..scanner_parameters import ScannerParameters


def New_minimal_echo_time(scanner_parameters: ScannerParameters):
    """
    :param scanner_parameters: Scan parameters
    :return: The minimal echo time in seconds
    """
    # extracting the scan parameters
    g_max = scanner_parameters.g_max
    t90 = scanner_parameters.t_90
    t180 = scanner_parameters.t_180
    s_max = scanner_parameters.s_max
    
    t_ramp = g_max / s_max  # ramp time
    
    te_min = t90 + t180 + 4 * t_ramp

    return te_min


def minimal_echo_time(b, scanner_parameters: ScannerParameters):
    """

    :param b: The bvalue in seconds/millimeters^2
    :param scanner_parameters: Scan parameters
    :return: The minimal echo time in seconds
    """
    # copying so we dont actually change bvalues that this function takes
    b = copy(b)

    # for the zero b-values we just take the b = 50 s/mm^2 since it will be a suitable constraint
    # Avoids singularity in minimal echo time computation
    b[b == 0] = 5.0 * unit('s/mm²')

    delta_max = compute_delta_max(b, scanner_parameters)
    domain = np.linspace(1e-9, delta_max, num=1000) * unit('s')
    te_min = np.min(echo_time(domain, b, scanner_parameters), axis=0)

    return te_min


def echo_time(delta, b, scanner_parameters: ScannerParameters):
    # extracting the scan parameters
    g_max = scanner_parameters.g_max
    t_rise = scanner_parameters.t_rise
    t90 = scanner_parameters.t_90
    t_half = scanner_parameters.t_half

    # surrogate parameter for readability
    b_g = b / (GAMMA ** 2 * g_max ** 2)

    Delta = (1 / delta ** 2) * (b_g - (1 / 30) * t_rise ** 3) + delta / 3 + (1 / (6 * delta)) * t_rise ** 2
    return 0.5 * t90 + Delta + delta + t_rise + t_half


def compute_delta_max(b, scanner_parameters: ScannerParameters):
    t180 = scanner_parameters.t_180
    t90 = scanner_parameters.t_90
    t_rise = scanner_parameters.t_rise
    t_half = scanner_parameters.t_half
    g_max = scanner_parameters.g_max

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
        # a1 = -t_rise / 4  # TODO: Units should be s²?
        a1 = -t_rise**2 / 4  # TODO: This is probably wrong, but a quick fix for the units.
        # a0 = (-3 / 2) * (b / (gamma_different_unit ** 2 * g_max ** 2)) + t_rise / 20  # TODO: Left side of the addition has units s³, whereas the right side s! In the echo_time function, t_rise³ is subtracted from B.
        a0 = (-3 / 2) * (b / (GAMMA ** 2 * g_max ** 2)) + t_rise**3 / 30  # TODO: This is probably wrong, but a quick fix for the units.
        return largest_real_cbrt(a2, a1, a0)

    def compute_delta_max_2():
        """
        0.5 TE == 0.5 t90 + 0.5 t180 + delta + trise

        :return: delta_max 2
        """
        T1 = 0.5 * t90 + 0.5 * t180 + t_rise
        T2 = 0.5 * t90 + t_rise + t_half

        B = b / (GAMMA ** 2 * g_max ** 2)  # TODO: B has units s³, but a few lines below are seconds subtracted. In the echo_time function, t_rise³ is subtracted from B.

        # TODO: this is the same as above, apart from t_rise/30 vs t_rise/20, which is a typo I think.
        a2 = -3 * (0.5 * T2 - T1)
        # a1 = -0.25 * t_rise  # TODO: Units should be s²?
        a1 = -0.25 * t_rise**2  # TODO: This is probably wrong, but a quick fix for the units.
        # a0 = -1.5 * (B - t_rise / 30)  # TODO: Units should be s³?
        a0 = -1.5 * (B - t_rise**3 / 30)  # TODO: This is probably wrong, but a quick fix for the units.
        return largest_real_cbrt(a2, a1, a0)

    delta_max_1 = compute_delta_max_1()
    delta_max_2 = compute_delta_max_2()
    delta_max = np.min(np.array([delta_max_1, delta_max_2]).T, axis=1)
    return delta_max

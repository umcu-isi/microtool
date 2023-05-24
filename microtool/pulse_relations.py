import numpy as np

from .constants import GAMMA
from .scanner_parameters import ScannerParameters

BVALUE_CONVERSION = 1e3  # ms/m^2 -> s/mm^2


def get_b_value_simplified(G, Delta, delta):
    """

    :param G: mT/m
    :param Delta: ms
    :param delta: ms
    :return: b value in s/mm^2
    """
    # unit track (no conversion): (1/ms . 1/mT)^2 * (mT/m)^2 * ms^2 * ms = ms/m^2
    return BVALUE_CONVERSION * GAMMA ** 2 * G ** 2 * delta ** 2 * (Delta - delta / 3)


def get_b_value_complete(G: np.ndarray, Delta: np.ndarray, delta: np.ndarray, scanner_parameters: ScannerParameters):
    """

    :param G: mT/m
    :param Delta: ms
    :param delta: ms
    :return: b value in s/mm^2
    """
    t_rise = compute_t_rise(G, scanner_parameters)
    prefactor = GAMMA ** 2 * G ** 2
    term1 = delta ** 2 * (Delta - delta / 3)
    term2 = (1 / 30) * t_rise ** 3
    term3 = - (delta / 6) * t_rise ** 2
    return BVALUE_CONVERSION * prefactor * (term1 + term2 + term3)


def compute_t_rise(G: np.ndarray, scanner_parameters: ScannerParameters):
    """

    :param G: mT/m
    :param scanner_parameters:
    :return:
    """

    # scanner_parameters.S_max has units mT / mm / ms
    s_max = scanner_parameters.S_max * 1e-3  # convert mT / mm / ms -> mT/m/ms
    return G / s_max

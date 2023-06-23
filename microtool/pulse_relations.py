import numpy as np

from .scanner_parameters import ScannerParameters


def get_b_value_simplified(gamma, G, Delta, delta):
    """
    :param G: T/m
    :param Delta: ms
    :param delta: ms
    :return: b value in s/mm^2
    """
    b_val = ((gamma * G * delta) ** 2) * (Delta - (delta / 3))
    return b_val


def get_b_value_complete(gamma, G: np.ndarray, Delta: np.ndarray, delta: np.ndarray,
                         scanner_parameters: ScannerParameters):
    """

    :param G: mT/mm
    :param Delta: ms
    :param delta: ms
    :return: b value in s/mm^2
    """
    t_rise = compute_t_rise(G, scanner_parameters)
    prefactor = gamma ** 2 * G ** 2
    term1 = delta ** 2 * (Delta - delta / 3)
    term2 = (1 / 30) * t_rise ** 3
    term3 = - (delta / 6) * t_rise ** 2
    return prefactor * (term1 + term2 + term3)


def compute_t_rise(G: np.ndarray, scanner_parameters: ScannerParameters):
    """

    :param G: mT/mm
    :param scanner_parameters:
    :return: The rise time in s
    """
    s_max = scanner_parameters.S_max  # mT/mm/ms
    tr_ms = G / s_max
    return tr_ms * 1e-3

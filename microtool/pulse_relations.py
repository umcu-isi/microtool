import numpy as np

from .scanner_parameters import ScannerParameters


# TODO update docstrings

def get_gradients(gamma, b, Delta, delta, scanner_parameters: ScannerParameters):
    """

    :param gamma: 1/mT/s
    :param b: b-value in s/mm²
    :param Delta: s
    :param delta: s
    :param scanner_parameters: scanner parameter definition
    :return: numpy array with gradient magnitudes
    """
    t_r = scanner_parameters.t_rise
    d = gamma ** 2 * (delta ** 2 * (Delta - delta / 3) + (1 / 30) * t_r ** 3 - (delta * t_r ** 2) / 6)
    return np.sqrt(b / d)


def get_b_value_simplified(gamma, G, Delta, delta):
    """
    :param G: T/m
    :param Delta: s
    :param delta: s
    :return: b value in s/mm^2
    """
    b_val = ((gamma * G * delta) ** 2) * (Delta - (delta / 3))
    return b_val


def get_b_value_complete(gamma, G: np.ndarray, Delta: np.ndarray, delta: np.ndarray,
                         scanner_parameters: ScannerParameters):
    """

    :param G: mT/mm
    :param Delta: s
    :param delta: s
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
    :param scanner_parameters: scanner parameter definition
    :return: The rise time in s
    """
    s_max = scanner_parameters.S_max  # mT/mm/ms
    tr_ms = G / s_max
    return tr_ms * 1e-3

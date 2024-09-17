from typing import Optional

import numpy as np

from .constants import GAMMA
from .scanner_parameters import ScannerParameters


# TODO update docstrings

def get_gradients(b, Delta, delta, scanner_parameters: ScannerParameters):
    """

    :param b: b-value in s/mm²
    :param Delta: s
    :param delta: s
    :param scanner_parameters: scanner parameter definition
    :return: numpy array with gradient magnitudes
    """
    t_r = scanner_parameters.t_rise
    d = GAMMA ** 2 * (delta ** 2 * (Delta - delta / 3) + (1 / 30) * t_r ** 3 - (delta * t_r ** 2) / 6)
    return np.sqrt(b / d)


def compute_b_values(gradient_magnitude: np.ndarray, pulse_interval: np.ndarray, pulse_width: np.ndarray,
                     scanner_parameters: Optional[ScannerParameters] = None):
    """
    Compute b-values from pulse intervals (∆) and pulse widths (δ). Optionally, scanner parameters can be provided for
     a more precise result.

    :param gradient_magnitude: Gradient magnitudes [mT/mm]
    :param pulse_interval: Pulse intervals [s]
    :param pulse_width: Pulse widths [s]
    :param scanner_parameters: scanner parameter definition
    :return: b-values [s/mm²]
    """
    if scanner_parameters is None:
        return ((GAMMA * gradient_magnitude * pulse_width)**2) * (pulse_interval - (pulse_width / 3))
    else:
        # See the 'Advanced Discussion' on https://mriquestions.com/what-is-the-b-value.html
        t_rise = compute_t_rise(gradient_magnitude, scanner_parameters)
        prefactor = (GAMMA * gradient_magnitude)**2
        term1 = pulse_width**2 * (pulse_interval - pulse_width / 3)
        term2 = t_rise**3 / 30
        term3 = -(pulse_width * t_rise**2) / 6
        return prefactor * (term1 + term2 + term3)


def compute_t_rise(g: np.ndarray, scanner_parameters: ScannerParameters):
    """

    :param g: mT/mm
    :param scanner_parameters: scanner parameter definition
    :return: The rise time in s
    """
    s_max = scanner_parameters.s_max  # Maximum slew rate in [mT/mm/s]
    return g / s_max

from typing import Optional

import numpy as np

from .constants import GAMMA
from .scanner_parameters import ScannerParameters


# TODO update docstrings

def get_gradients(b, pulse_duration, pulse_interval, scanner_parameters: ScannerParameters):
    """

    :param b: b-value in s/mm²
    :param pulse_duration: s
    :param pulse_interval: s
    :param scanner_parameters: scanner parameter definition
    :return: numpy array with gradient magnitudes
    """
    t_r = scanner_parameters.t_rise
    d = GAMMA ** 2 * (pulse_duration ** 2 * (pulse_interval - pulse_duration / 3) + (1 / 30) * t_r ** 3 - (pulse_duration * t_r ** 2) / 6)
    return np.sqrt(b / d)


def compute_b_values(pulse_duration: np.ndarray, pulse_interval: np.ndarray, pulse_magnitude: np.ndarray,
                     scanner_parameters: Optional[ScannerParameters] = None):
    """
    Compute b-values from pulse intervals (∆) and pulse durations (δ). Optionally, scanner parameters can be provided for
     a more precise result.

    :param pulse_duration: Pulse durations [s]
    :param pulse_interval: Pulse intervals [s]
    :param pulse_magnitude: Pulse magnitudes [mT/mm]
    :param scanner_parameters: scanner parameter definition
    :return: b-values [s/mm²]
    """
    if scanner_parameters is None:
        return ((GAMMA * pulse_magnitude * pulse_duration)**2) * (pulse_interval - (pulse_duration / 3))
    else:
        # See the 'Advanced Discussion' on https://mriquestions.com/what-is-the-b-value.html
        t_rise = compute_t_rise(pulse_magnitude, scanner_parameters)
        prefactor = (GAMMA * pulse_magnitude)**2
        term1 = pulse_duration**2 * (pulse_interval - pulse_duration / 3)
        term2 = t_rise**3 / 30
        term3 = -(pulse_duration * t_rise**2) / 6
        return prefactor * (term1 + term2 + term3)


def compute_t_rise(pulse_magnitude: np.ndarray, scanner_parameters: ScannerParameters):
    """

    :param pulse_magnitude: mT/mm
    :param scanner_parameters: scanner parameter definition
    :return: The rise time in s
    """
    s_max = scanner_parameters.s_max  # Maximum slew rate in [mT/mm/s]
    return pulse_magnitude / s_max

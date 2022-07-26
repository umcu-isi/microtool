"""
What scheme is flavius using?
"""

from typing import List, Union, Dict

import numpy as np

from microtool.acquisition_scheme import AcquisitionScheme, AcquisitionParameters
from scipy.optimize import LinearConstraint


class FlaviusAcquisitionScheme(AcquisitionScheme):
    """

    :param b_values:
    :param echo_times:
    :param max_gradient:
    """

    def __init__(self, b_values: Union[List[float], np.ndarray], echo_times: Union[List[float], np.ndarray],
                 max_gradient: np.ndarray,
                 max_slew_rate: np.ndarray,
                 half_readout_time: np.ndarray,
                 excitation_time_pi: np.ndarray,
                 excitation_time_halfpi: np.ndarray
                 ):
        # Check for b0 values? make sure initial scheme satisfies constraints.

        super().__init__({
            'DiffusionBvalue': AcquisitionParameters(
                values=b_values, unit='s/mm^2', scale=1000, lower_bound=0.0, upper_bound=3e4
            ),
            'EchoTime': AcquisitionParameters(
                values=echo_times, unit='ms', scale=10, lower_bound=.1, upper_bound=1e3
            ),
            'MaxPulseGradient': AcquisitionParameters(
                values=max_gradient, unit='mT/m', scale=10, fixed=True
            ),
            'MaxSlewRate': AcquisitionParameters(
                values=max_slew_rate, unit='mT/m/ms', scale=10, fixed=True
            ),
            'HalfReadTime': AcquisitionParameters(
                values=half_readout_time, unit='ms', scale=10, fixed=True
            ),
            'PulseDurationPi': AcquisitionParameters(
                values=excitation_time_pi, unit='ms', scale=10, fixed=True
            ),
            'PulseDurationHalfPi': AcquisitionParameters(
                values=excitation_time_halfpi, unit='ms', scale=10, fixed=True
            )
        })

    @property
    def echo_times(self):
        return self['EchoTime'].values

    @property
    def b_values(self):
        return self['DiffusionBvalue'].values

    @property
    def max_gradient(self):
        return self['MaxPulseGradient'].values

    def get_constraints(self) -> Union[dict, List[dict]]:
        raise NotImplementedError()


def compute_echo_times(b_vals, slew_time, max_gradient):
    """
    Computing the echo times as proposed by the script that chantal provided.

    :param b_vals:
    :param slew_time:
    :param max_gradient:
    :return:
    """
    raise NotImplementedError()

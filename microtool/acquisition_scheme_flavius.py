"""
What scheme is flavius using?
"""

from typing import List, Union

import numpy as np

from microtool.acquisition_scheme import AcquisitionScheme, AcquisitionParameters
from microtool.utils.solve_echo_time import minimal_echo_time


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
                 excitation_time_half_pi: np.ndarray
                 ):
        # Check for b0 values? make sure initial scheme satisfies constraints.

        super().__init__({
            'DiffusionBvalue': AcquisitionParameters(
                values=b_values, unit='s/mm^2', scale=1000, lower_bound=0.0, upper_bound=3e4
            ),
            'EchoTime': AcquisitionParameters(
                values=echo_times, unit='ms', scale=10, lower_bound=.1, upper_bound=2e2
            ),
            'MaxPulseGradient': AcquisitionParameters(
                values=max_gradient, unit='mT/mm', scale=1, fixed=True
            ),
            'MaxSlewRate': AcquisitionParameters(
                values=max_slew_rate, unit='mT/mm/ms', scale=1, fixed=True
            ),
            'RiseTime': AcquisitionParameters(
                values=max_gradient / max_slew_rate, unit='ms', scale=1, fixed=True
            ),
            'HalfReadTime': AcquisitionParameters(
                values=half_readout_time, unit='ms', scale=10, fixed=True
            ),
            'PulseDurationPi': AcquisitionParameters(
                values=excitation_time_pi, unit='ms', scale=10, fixed=True
            ),
            'PulseDurationHalfPi': AcquisitionParameters(
                values=excitation_time_half_pi, unit='ms', scale=10, fixed=True
            )
        })

    @property
    def echo_times(self):
        return self['EchoTime'].values

    @property
    def b_values(self):
        return self['DiffusionBvalue'].values

    def get_constraints(self) -> Union[dict, List[dict]]:
        t180 = self['PulseDurationPi'].values
        t90 = self['PulseDurationHalfPi'].values
        G_max = self['MaxPulseGradient'].values
        t_rise = self['RiseTime'].values
        t_half = self['HalfReadTime'].values

        def fun(x: np.ndarray) -> np.ndarray:
            # get b-values from x
            b = self.get_parameter_from_parameter_vector('DiffusionBvalue', x)
            # note that b is in s/mm^2 but all other time dimensions are ms.
            # so we convert to ms/mm^2
            b *= 1e3
            # get echotimes from x, (units are # ms)
            TE = self.get_parameter_from_parameter_vector('EchoTime', x)
            # compute the minimal echotimes associated with b-values and other parameters
            TE_min = minimal_echo_time(b, t90, t180, t_half, G_max, t_rise)

            # The constraint is satisfied if actual TE is higher than minimal TE
            return TE - TE_min

        return {'type': 'ineq', 'fun': fun}

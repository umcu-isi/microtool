"""
What scheme is flavius using?
"""


from typing import List, Union

import numpy as np

from microtool.acquisition_scheme import AcquisitionScheme, AcquisitionParameters
from scipy.optimize import LinearConstraint

class FlaviusAcquisitionScheme(AcquisitionScheme):

    def __init__(self, b_values: Union[List[float], np.ndarray], echo_times: Union[List[float], np.ndarray]):
        # Check on values etc etc

        # Call the superclass constructor with correct optimization parameters

        super().__init__({
            'DiffusionBvalue': AcquisitionParameters(
                values=b_values, unit='s/mm^2', scale=1000, lower_bound=0.0, upper_bound=3e4
            ),
            'EchoTime': AcquisitionParameters(
                values=echo_times, unit='ms', scale=10, lower_bound=.1, upper_bound=1e3
            )
        })

    @property
    def echo_times(self):
        return self['EchoTime'].values

    @property
    def b_values(self):
        return self['DiffusionBvalue'].values

    def get_constraints(self) -> LinearConstraint:
        return None
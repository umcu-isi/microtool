from dataclasses import dataclass
from typing import Optional

import numpy as np


class DiffusionModel:
    """
    Base class for MR tissue diffusion models. Derived classes must implement 1) a __call__ function that returns the
    MR signal attenuation and 2) a function that returns the derivatives of the signal attenuation with respect to the
    tissue parameters.
    """
    def __call__(self,
                 b_values: np.ndarray,
                 b_vectors: np.ndarray,
                 pulse_widths: np.ndarray,
                 pulse_intervals: np.ndarray) -> np.ndarray:
        """
        Calculates the MR signal attenuation given b-values, b-vectors, pulse widths and pulse intervals.

        :param b_values: A numpy array of b-values in s/mm².
        :param b_vectors: A numpy array of direction cosines.
        :param pulse_widths: A numpy array of pulse widths δ in seconds.
        :param pulse_intervals: A numpy array of pulse intervals Δ in seconds.
        :return: An array with signal attenuation values.
        """
        raise NotImplementedError()

    def jacobian(self,
                 b_values: np.ndarray,
                 b_vectors: np.ndarray,
                 pulse_widths: np.ndarray,
                 pulse_intervals: np.ndarray) -> np.ndarray:
        """
        Calculates the change in MR signal attenuation due to a change in the tissue model parameters.

        :param b_values: A numpy array of b-values in s/mm².
        :param b_vectors: A numpy array of direction cosines.
        :param pulse_widths: A numpy array of pulse widths δ in seconds.
        :param pulse_intervals: A numpy array of pulse intervals Δ in seconds.
        :return: An N×M Jacobian matrix, where N is the number of samples and M is the number of parameters.
        """
        raise NotImplementedError()

    def get_scales(self) -> np.ndarray:
        """
        Returns the parameter scales of the diffusion parameters.

        :return: An array with parameter scales.
        """
        raise NotImplementedError()


@dataclass
class TissueModel:
    """
    Defines a tissue by T1 and T2 relaxation parameters and/or an MR diffusion model.
    """
    t1: Optional[float] = None
    t2: Optional[float] = None
    diffusion_model: Optional[DiffusionModel] = None

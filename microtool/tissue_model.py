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


# TODO: Take T2* and relaxation parameter distributions into account. See eq. 5 and 6 in
#  https://www.ncbi.nlm.nih.gov/books/NBK567564/
# TODO: Specify fixed tissue parameters? Or let the AcquisitionScheme deal with that?
@dataclass
class TissueModel:
    # noinspection PyUnresolvedReferences
    """
    Defines a tissue by relaxation parameters and/or an MR diffusion model.

    :param s0: MR signal from fully recovered magnetisation, just before the 90° RF pulse.
    :param t1: Longitudinal relaxation time constant T1 in seconds.
    :param t2: Transverse relaxation time constant T2 in seconds.
    :param diffusion_model: A DiffusionModel that describes diffusion MR tissue properties.
    """
    s0: float = 1.0
    t1: Optional[float] = None
    t2: Optional[float] = None
    diffusion_model: Optional[DiffusionModel] = None

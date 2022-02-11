from typing import Union, List, Tuple

import numpy as np


class AcquisitionScheme:
    def __init__(self,
                 b_values: Union[List[float], np.ndarray],
                 b_vectors: Union[List[Tuple[float, float, float]], np.ndarray],
                 pulse_widths: Union[List[float], np.ndarray],
                 pulse_intervals: Union[List[float], np.ndarray]):
        """
        Creates a MICROtool acquisition scheme.

        :param b_values: A list or numpy array of B-values in s/mm².
        :param b_vectors: A list or numpy array of direction cosines.
        :param pulse_widths: A list or numpy array of pulse widths in seconds.
        :param pulse_intervals: A list or numpy array of pulse intervals in seconds.
        """
        # Copy the input.
        b_values = np.asarray(b_values, dtype=np.float64)
        b_vectors = np.asarray(b_vectors, dtype=np.float64)
        pulse_widths = np.asarray(pulse_widths, dtype=np.float64)
        pulse_intervals = np.asarray(pulse_intervals, dtype=np.float64)

        # Normalize the gradient directions and calculate the spherical coordinates.
        b_vectors *= 1 / np.linalg.norm(b_vectors, axis=1).reshape(-1, 1)
        phi = np.arctan2(b_vectors[:, 1], b_vectors[:, 0])
        theta = np.arccos(b_vectors[:, 2])

        # Construct an N×5 matrix of acquisition parameters.
        self._parameters = np.array([b_values, phi, theta, pulse_widths, pulse_intervals]).T

        # Initial active parameter mask.
        # TODO: For starters, we're only marking only b-values as 'active'.
        self._mask = np.zeros_like(self._parameters, dtype=bool)
        self._mask[:, 0] = True

    def get_b_values(self) -> np.ndarray:
        return self._parameters[:, 0].copy()

    def get_phi(self) -> np.ndarray:
        return self._parameters[:, 1].copy()

    def get_theta(self) -> np.ndarray:
        return self._parameters[:, 2].copy()

    def get_pulse_widths(self) -> np.ndarray:
        return self._parameters[:, 3].copy()

    def get_pulse_intervals(self) -> np.ndarray:
        return self._parameters[:, 4].copy()

    def get_b_vectors(self) -> np.ndarray:
        phi = self._parameters[:, 1]
        theta = self._parameters[:, 2]
        sin_theta = np.sin(theta)
        return np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), np.cos(theta)]).T

"""
Insert useful information
"""
import numpy as np
import pytest

from microtool.acquisition_scheme import DiffusionAcquisitionScheme
from microtool.gradient_sampling import sample_uniform


class TestDiffusionAcquisitionScheme:
    def test_b_values(self):
        """
        Checking if we come up with the same b_values as alexander given the other pulse parameters
        """
        expected_b_values = [20087, 18771, 6035, 1744]
        M = 4  # number of parameter combinations for every direction

        gradient_vectors = sample_uniform(M)

        gradient_magnitudes = np.array([.200, .200, .121, .200])
        Delta = np.array([25.0, 26.0, 29.0, 13.0]) * 1e-3
        delta = np.array([20.0, 18.0, 16.0, 8.0]) * 1e-3

        scheme = DiffusionAcquisitionScheme(gradient_vectors, gradient_magnitudes, delta, Delta,
                                            echo_times=np.repeat(80.0e-3, M))

        assert scheme.b_values == pytest.approx(expected_b_values, rel=0.1)

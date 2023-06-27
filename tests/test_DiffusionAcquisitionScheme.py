"""
Insert useful information
"""
import numpy as np
import pytest

from microtool.acquisition_scheme import DiffusionAcquisitionScheme
from microtool.gradient_sampling import sample_uniform


class TestDiffusionAcquisitionScheme:
    expected_b_values = [20087, 18771, 6035, 1744]
    M = 4  # number of parameter combinations for every direction

    gradient_vectors = sample_uniform(M)

    gradient_magnitudes = np.array([.200, .200, .121, .200])
    Delta = np.array([25.0, 26.0, 29.0, 13.0]) * 1e-3
    delta = np.array([20.0, 18.0, 16.0, 8.0]) * 1e-3

    def test_default_constructor(self):
        """
        Checking if we come up with the same b_values as alexander given the other pulse parameters
        """

        scheme = DiffusionAcquisitionScheme(self.gradient_vectors, self.gradient_magnitudes, self.delta, self.Delta,
                                            echo_times=np.repeat(80.0e-3, self.M))

        assert scheme.b_values == pytest.approx(self.expected_b_values, rel=0.1)

    def test_b_value_constructor(self):
        scheme = DiffusionAcquisitionScheme.from_bvals(self.expected_b_values, self.gradient_vectors, self.Delta,
                                                       self.delta)
        assert scheme.pulse_magnitude == pytest.approx(self.gradient_magnitudes, rel=.1)

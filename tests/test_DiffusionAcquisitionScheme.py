"""
Insert useful information
"""
import matplotlib.pyplot as plt
import numpy as np
import pytest
from dmipy.signal_models.sphere_models import S2SphereStejskalTannerApproximation

from microtool.acquisition_scheme import DiffusionAcquisitionScheme
from microtool.dmipy import make_microtool_tissue_model
from microtool.gradient_sampling import sample_uniform
from microtool.optimize import optimize_scheme
from microtool.utils.plotting import plot_acquisition_parameters, LossInspector


class TestDiffusionAcquisitionSchemeConstruction:
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

        scheme = DiffusionAcquisitionScheme(self.gradient_magnitudes, self.gradient_vectors, self.delta, self.Delta)
        assert scheme.b_values == pytest.approx(self.expected_b_values, rel=0.1)

    def test_b_value_constructor(self):
        scheme = DiffusionAcquisitionScheme.from_bvals(self.expected_b_values, self.gradient_vectors, self.delta,
                                                       self.Delta)
        assert scheme.pulse_magnitude == pytest.approx(self.gradient_magnitudes, rel=.1)


class TestDiffusionAcquisitionSchemeMethods:
    magnitudes = np.array([0, 0.2])
    directions = np.array([[0, 0, 1], [0, 1, 0]])
    pulse_widths = np.array([0.007, 0.007])
    pulse_intervals = np.array([0.035, 0.035])
    scheme = DiffusionAcquisitionScheme(magnitudes, directions, pulse_widths=pulse_widths,
                                        pulse_intervals=pulse_intervals)

    def test_fix_b0_measurements(self):
        """
        We test if using the fix b0 measurement method correctly supplements the existing fixed masks
        """
        first_fix_mask = np.array([False, True])
        self.scheme["DiffusionPulseWidth"].set_fixed_mask(first_fix_mask)

        self.scheme.fix_b0_measurements()
        where_b0 = self.scheme.b_values == 0

        # The expected fixed mask for diffusion pulse width is now determined by both the fixation we did earlier and
        # the b0 measurements that need to be fixed
        expected_mask = np.logical_or(where_b0, first_fix_mask)

        actual_mask = np.logical_not(self.scheme["DiffusionPulseWidth"].optimize_mask)
        np.testing.assert_equal(expected_mask, actual_mask)

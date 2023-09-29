"""
Insert useful information
"""
import numpy as np
import pytest

from microtool.acquisition_scheme import DiffusionAcquisitionScheme
from microtool.gradient_sampling import sample_uniform
from microtool.scanner_parameters import ScannerParameters


class TestDiffusionAcquisitionSchemeConstruction:
    expected_b_values = [20087, 18771, 6035, 1744]
    M = 4  # number of parameter combinations for every direction

    gradient_vectors = sample_uniform(M)

    gradient_magnitudes = np.array([.200, .200, .121, .200])
    Delta = np.array([25.0, 26.0, 29.0, 13.0]) * 1e-3
    delta = np.array([20.0, 18.0, 16.0, 8.0]) * 1e-3
    scan_parameters = ScannerParameters(4.e-3, 4.e-3, 14.e-3, 400e-3, np.inf)

    def test_default_constructor(self):
        """
        Checking if we come up with the same b_values as alexander given the other pulse parameters
        """

        scheme = DiffusionAcquisitionScheme(self.gradient_magnitudes, self.gradient_vectors, self.delta, self.Delta,
                                            scan_parameters=self.scan_parameters)
        assert scheme.b_values == pytest.approx(self.expected_b_values, rel=0.1)

    def test_b_value_constructor(self):
        scheme = DiffusionAcquisitionScheme.from_bvals(self.expected_b_values, self.gradient_vectors, self.delta,
                                                       self.Delta, scan_parameters=self.scan_parameters)
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

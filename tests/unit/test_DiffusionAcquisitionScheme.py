"""
Insert useful information
"""
import numpy as np

from microtool.acquisition_scheme import DiffusionAcquisitionScheme
from microtool.gradient_sampling import sample_uniform
from microtool.scanner_parameters import ScannerParameters
from microtool.utils.unit_registry import unit


class TestDiffusionAcquisitionSchemeConstruction:
    expected_b_values = np.array([20087, 18771, 6035, 1744]) * unit('s/mm²')
    M = 4  # number of parameter combinations for every direction

    gradient_vectors = sample_uniform(M)

    pulse_magnitudes = np.array([.200, .200, .121, .200]) * unit('mT/mm')
    pulse_interval = np.array([25.0, 26.0, 29.0, 13.0]) * 1e-3 * unit('s')
    pulse_duration = np.array([20.0, 18.0, 16.0, 8.0]) * 1e-3 * unit('s')
    scanner_parameters = ScannerParameters(
        4.e-3 * unit('s'),
        4.e-3 * unit('s'),
        14.e-3 * unit('s'),
        400e-3 * unit('mT/mm'),
        1300 * unit('mT/mm/s'))

    def test_default_constructor(self):
        """
        Checking if we come up with the same b_values as alexander given the other pulse parameters
        """

        scheme = DiffusionAcquisitionScheme(self.pulse_magnitudes, self.gradient_vectors, self.pulse_duration,
                                            self.pulse_interval, scanner_parameters=self.scanner_parameters)
        assert np.allclose(scheme.b_values, self.expected_b_values, rtol=0.1)

    def test_b_value_constructor(self):
        scheme = DiffusionAcquisitionScheme.from_bvals(self.expected_b_values, self.gradient_vectors,
                                                       self.pulse_duration, self.pulse_interval,
                                                       scanner_parameters=self.scanner_parameters)
        assert np.allclose(scheme.pulse_magnitude, self.pulse_magnitudes, rtol=0.1)


class TestDiffusionAcquisitionSchemeMethods:
    magnitudes = np.array([0, 0.2]) * unit('mT/mm')
    directions = np.array([[0, 0, 1], [0, 1, 0]])
    pulse_durations = np.array([0.007, 0.007]) * unit('s')
    pulse_intervals = np.array([0.035, 0.035]) * unit('s')
    scheme = DiffusionAcquisitionScheme(magnitudes, directions, pulse_durations=pulse_durations,
                                        pulse_intervals=pulse_intervals)

    def test_fix_b0_measurements(self):
        """
        We test if using the fix b0 measurement method correctly supplements the existing fixed masks
        """
        first_fix_mask = np.array([False, True])
        self.scheme["DiffusionPulseDuration"].set_fixed_mask(first_fix_mask)

        self.scheme.fix_b0_measurements()
        where_b0 = self.scheme.b_values == 0

        # The expected fixed mask for diffusion pulse duration is now determined by both the fixation we did earlier and
        # the b0 measurements that need to be fixed
        expected_mask = np.logical_or(where_b0, first_fix_mask)

        actual_mask = np.logical_not(self.scheme["DiffusionPulseDuration"].optimize_mask)
        np.testing.assert_equal(expected_mask, actual_mask)

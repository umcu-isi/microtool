"""
Tests for DiffusionAcquisitionScheme_OGSE construction and methods.
"""

import numpy as np

from microtool.acquisition_scheme import DiffusionAcquisitionScheme_OGSE
from microtool.gradient_sampling import sample_uniform
from microtool.scanner_parameters import ScannerParameters
from microtool.utils.unit_registry import unit


class TestDiffusionAcquisitionSchemeOGSEConstruction:
    M = 4

    gradient_vectors = sample_uniform(M)
    n_periods = np.array([1, 1, 1, 1])

    pulse_magnitudes = np.array([0.200, 0.200, 0.121, 0.200]) * unit("mT/mm")
    pulse_durations = np.array([20.0, 18.0, 16.0, 8.0]) * 1e-3 * unit("s")
    pulse_intervals = np.array([25.0, 26.0, 29.0, 13.0]) * 1e-3 * unit("s")

    scanner_parameters = ScannerParameters(
        4e-3 * unit("s"),
        4e-3 * unit("s"),
        14e-3 * unit("s"),
        400e-3 * unit("mT/mm"),
        1300 * unit("mT/mm/s"),
    )

    def test_default_constructor(self):
        """Check that the OGSE scheme calculates b-values."""
        scheme = DiffusionAcquisitionScheme_OGSE(
            n_periods=self.n_periods,
            pulse_magnitude=self.pulse_magnitudes,
            gradient_directions=self.gradient_vectors,
            pulse_durations=self.pulse_durations,
            pulse_intervals=self.pulse_intervals,
            scanner_parameters=self.scanner_parameters,
        )

        assert np.all(np.isfinite(scheme.b_values))
        assert np.all(scheme.b_values >= 0)

    def test_b_value_constructor(self):
        """Check that construction from b-values reproduces the requested values."""
        scheme = DiffusionAcquisitionScheme_OGSE(
            n_periods=self.n_periods,
            pulse_magnitude=self.pulse_magnitudes,
            gradient_directions=self.gradient_vectors,
            pulse_durations=self.pulse_durations,
            pulse_intervals=self.pulse_intervals,
            scanner_parameters=self.scanner_parameters,
        )

        target_b_values = scheme.b_values

        reconstructed = DiffusionAcquisitionScheme_OGSE.from_bvals(
            target_b_values,
            self.gradient_vectors,
            self.pulse_durations,
            self.pulse_intervals,
            scanner_parameters=self.scanner_parameters,
        )

        np.testing.assert_allclose(
            reconstructed.b_values,
            target_b_values,
            rtol=0.1,
        )


class TestDiffusionAcquisitionSchemeOGSEMethods:
    n_periods = np.array([1, 1])

    magnitudes = np.array([0.0, 0.2]) * unit("mT/mm")
    directions = np.array([
        [0, 0, 1],
        [0, 1, 0],
    ])

    pulse_durations = np.array([0.007, 0.007]) * unit("s")
    pulse_intervals = np.array([0.035, 0.035]) * unit("s")

    scheme = DiffusionAcquisitionScheme_OGSE(
        n_periods=n_periods,
        pulse_magnitude=magnitudes,
        gradient_directions=directions,
        pulse_durations=pulse_durations,
        pulse_intervals=pulse_intervals,
    )

    def test_fix_b0_measurements(self):
        """Check that fixing b0 measurements preserves existing fixed masks."""
        first_fix_mask = np.array([False, True])

        self.scheme["DiffusionPulseDuration"].set_fixed_mask(
            first_fix_mask
        )

        self.scheme.fix_b0_measurements()

        where_b0 = self.scheme.b_values == 0

        expected_mask = np.logical_or(
            where_b0,
            first_fix_mask,
        )

        actual_mask = np.logical_not(
            self.scheme["DiffusionPulseDuration"].optimize_mask
        )

        np.testing.assert_equal(expected_mask, actual_mask)

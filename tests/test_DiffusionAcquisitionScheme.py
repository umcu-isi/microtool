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


class TestDiffusionAcquisitionSchemeOptimization:
    model = make_microtool_tissue_model(S2SphereStejskalTannerApproximation(diameter=1e-6))

    directions = sample_uniform(5)
    gradient_magnitudes = np.array([0.0, .200, .200, .121, .200])
    Delta = np.array([25.0, 25.0, 26.0, 29.0, 13.0]) * 1e-3
    delta = np.array([20.0, 20.0, 5.0, 16.0, 8.0]) * 1e-3
    scheme = DiffusionAcquisitionScheme(gradient_magnitudes, directions, delta, Delta)
    scheme.fix_b0_measurements()

    def test_loss_landscape(self):
        loss_inspector = LossInspector(self.scheme, self.model, noise_var=.02)
        loss_inspector.plot([{"DiffusionPulseWidth": 1}])
        plt.savefig("loss_landscape.png")

    def test_optimize_scheme(self):
        print(self.scheme)
        plot_acquisition_parameters(self.scheme)
        plt.savefig("initial_scheme.png")
        plt.close()

        optimal_scheme, result = optimize_scheme(self.scheme, self.model, noise_variance=.02,
                                                 method="differential_evolution", solver_options={})

        print(optimal_scheme)
        plot_acquisition_parameters(optimal_scheme)
        plt.savefig("optimal_scheme.png")

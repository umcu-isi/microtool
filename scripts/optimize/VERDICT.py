"""
Script demonstrating the optimization of VERDICT tissue model. For now we are using an initial scheme from
DOI:10.1002/nbm.4019
"""

import matplotlib.pyplot as plt
import numpy as np
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.signal_models import sphere_models, cylinder_models, gaussian_models

from microtool.dmipy import convert_dmipy_scheme2diffusion_scheme, DmipyTissueModel
from microtool.gradient_sampling.uniform import sample_uniform
from microtool.optimize import optimize_scheme
from microtool.utils import plotting, IO


def main():
    # ------------------- Tissue model
    # The three models we will be using to model intra cellular, extracellular and Vascular signal respectively
    # Initialize diameter of the cell i.e. sphere in the middle of optimization domain
    sphere = sphere_models.S4SphereGaussianPhaseApproximation(diffusion_constant=2.0e-9, diameter=10e-6)
    ball = gaussian_models.G1Ball(lambda_iso=2.0e-9)
    stick = cylinder_models.C1Stick(mu=[np.pi / 2, np.pi / 2], lambda_par=8.0e-9)

    verdict_model = MultiCompartmentModel(models=[sphere, ball, stick])

    # In Bonet-Carne et. al. the diameter of the intracellular sphere (d_IC) and the vascular diffusivity (P)
    # can be fixed such that we can estimate d_{EES}
    verdict_model.set_fixed_parameter('S4SphereGaussianPhaseApproximation_1_diameter', 2.0e-9)
    verdict_model.set_fixed_parameter('C1Stick_1_lambda_par', 8.0e-9)

    # Really the orientations should also be fixed?
    verdict_model.set_fixed_parameter('C1Stick_1_mu', [np.pi / 2, np.pi / 2])
    verdict_model = DmipyTissueModel(verdict_model, [.3, .3, .4])

    IO.save_pickle(verdict_model, 'schemes/verdict_gt.pkl')
    print("The verdict tissue model:", verdict_model)

    # ------------------- Acquisition Scheme
    b_values = np.array([90, 500, 1500, 2000, 3000]) * 1e6
    TE = np.array([50, 65, 90, 71, 80]) * 1e-3
    gradient_directions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    Delta = np.array([23.8, 31.3, 43.8, 34.3, 38.8]) * 1e-3
    delta = np.array([3.9, 11.4, 23.9, 14.4, 18.9]) * 1e-3

    # Extending the arrays so dmipy understands what we mean
    n_measurements = len(b_values)
    n_directions = len(gradient_directions)

    # Executing a b0 measurement for every delta,Delta and TE combination (i.e. every b-value) so we can compensate
    # for T1 and T2 dependence.
    n_b0 = len(delta)
    b0 = np.zeros(n_b0)
    zero_directions = sample_uniform(n_b0)
    zero_delta = delta
    zero_Delta = Delta
    zero_TE = TE

    # Converting to dmipy compatible ,i.e., prepending b0 values and repeating for every direction
    b_values = convert_to_dmipy_acquisition_parameter(b0, b_values, n_directions)
    delta = convert_to_dmipy_acquisition_parameter(zero_delta, delta, n_directions)
    Delta = convert_to_dmipy_acquisition_parameter(zero_Delta, Delta, n_directions)
    TE = convert_to_dmipy_acquisition_parameter(zero_TE, TE, n_directions)

    # Extending directions for every measurement
    gradient_directions = np.tile(sample_uniform(n_directions), (n_measurements, 1))

    # Prepending b0 measurements
    gradient_directions = np.concatenate([zero_directions, gradient_directions], axis=0)
    scheme = acquisition_scheme_from_bvalues(b_values, gradient_directions, delta, Delta, TE=TE)
    scheme = convert_dmipy_scheme2diffusion_scheme(scheme)
    # For dmipy based optimization fixed b0 measurements are required.
    scheme["DiffusionBValue"].set_fixed_mask(scheme.b_values == 0)
    scheme['DiffusionPulseWidth'].fixed = True
    scheme['DiffusionPulseInterval'].fixed = True
    print(scheme)

    # -------------- Optimization
    trust_options = {'verbose': 2}
    best_scheme, opt_result = optimize_scheme(scheme, verdict_model, 0.02, method='trust-constr',
                                              solver_options=trust_options)

    IO.save_pickle(best_scheme, "schemes/verdict_optimal.pkl")
    IO.save_pickle(scheme, "schemes/verdict_start.pkl")

    # --------------- Generating figures
    optimized_parameter_fig = plotting.plot_acquisition_parameters(best_scheme, "Acquisition Parameters")
    plt.savefig("optimized_parameter.png")
    init_parameter_fig = plotting.plot_acquisition_parameters(scheme, "Acquisition Parameters")
    plt.savefig("init_parameter.png")
    fig_init_signal = plotting.plot_signal(scheme, verdict_model)
    plt.savefig("init_signal.png")
    fig_optimized_signal = plotting.plot_signal(best_scheme, verdict_model)
    plt.savefig("optimized_signal.png")
    plt.show()


def convert_to_dmipy_acquisition_parameter(par_b0, par_measurements, n_directions):
    n_measurements = len(par_measurements)

    # repeating the measurement for every direction
    par_repeated = np.repeat(par_measurements, n_directions)
    # prepending the b0 measurement of this parameter
    return np.concatenate([par_b0, par_repeated])


if __name__ == '__main__':
    main()

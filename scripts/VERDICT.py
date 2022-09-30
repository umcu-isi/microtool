"""
Script demonstrating the optimization of VERDICT tissue model. For now we are using an initial scheme from
DOI:10.1002/nbm.4019
"""
import matplotlib.pyplot as plt
import numpy as np
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.signal_models import sphere_models, cylinder_models, gaussian_models
from microtool.utils.gradient_sampling.uniform import sample_uniform
from microtool.dmipy import DmipyAcquisitionSchemeWrapper, DmipyTissueModel
from microtool.optimize import optimize_scheme


def main():
    # ------------------- Tissue model
    # The three models we will be using to model intra cellular, extra cellular and Vascular signal respectively
    # Initialize diameter of the cell i.e. sphere in the middle of optimization domain
    sphere = sphere_models.S4SphereGaussianPhaseApproximation(diffusion_constant=0.9e-9, diameter=10e-6)
    ball = gaussian_models.G1Ball(lambda_iso=0.9e-9)
    stick = cylinder_models.C1Stick(mu=[np.pi/2, np.pi/2], lambda_par=6.5e-9)

    verdict_model = MultiCompartmentModel(models=[sphere, ball, stick])

    print('The multicompartment model has the following parameters', verdict_model.parameter_names)

    # We fix the extra cellular diffusivuty in accordance with panagiotakis black magic number from 2014
    verdict_model.set_fixed_parameter('G1Ball_1_lambda_iso', 0.9e-9)
    # We set the optimization such that the pseudo diffusioin coefficient is larger than 3.05 um^2/ms
    verdict_model.set_parameter_optimization_bounds('C1Stick_1_lambda_par', [3.05e-9, 10e-9])

    verdict_model = DmipyTissueModel(verdict_model, np.array([.3, .3, .4]))
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
    scheme.print_acquisition_info
    scheme = DmipyAcquisitionSchemeWrapper(scheme)

    print(scheme)



    # -------------- Optimization
    best_scheme, _ = optimize_scheme(scheme, verdict_model, 0.02)

    # Signal plotting
    plt.figure('Signal plot')
    plt.plot(verdict_model(scheme), '.')
    plt.xlabel('Measurement')
    plt.xticks(range(scheme.pulse_count), np.array(range(scheme.pulse_count)) + 1)
    plt.ylabel(r'$S/S_0$')
    plt.show()


def convert_to_dmipy_acquisition_parameter(par_b0, par_measurements, n_directions):
    n_measurements = len(par_measurements)

    # repeating the measurement for every direction
    par_repeated = np.repeat(par_measurements, n_directions)
    # prepending the b0 measurement of this parameter
    return np.concatenate([par_b0, par_repeated])


if __name__ == '__main__':
    main()

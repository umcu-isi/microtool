from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues


from microtool.utils.gradient_sampling import sample_uniform

import numpy as np
import pprint


def main():
    # Alexander uses 4 measurements for every gradient direction. The combinations of gradients and pulse timings per
    # measurement are the same for all directions (they are allowed to be different between measurements but not
    # (directions)

    # setting some random b0 measurements because alexander does not specify.
    n_b0 = 18
    b0 = np.zeros(n_b0)
    zero_directions = sample_uniform(n_b0)
    zero_delta = np.repeat(0.007, n_b0)
    zero_Delta = np.repeat(0.012, n_b0)

    n_measurements = 4
    n_directions = 30
    # Extending b_values array for every direction

    bvalues = np.repeat(np.array([17370, 3580, 1216, 1205])* 1e6, n_directions)
    delta = np.repeat(np.array([0.019, 0.016, 0.007, 0.007]), n_directions)
    Delta = np.repeat(np.array([0.024, 0.027, 0.012, 0.012]), n_directions)
    gradient_directions = np.tile(sample_uniform(n_directions), (n_measurements, 1))

    # Prepending b0 measurements
    bvalues = np.concatenate([b0, bvalues])
    delta = np.concatenate([zero_delta, delta])
    Delta = np.concatenate([zero_Delta, Delta])
    gradient_directions = np.concatenate([zero_directions,gradient_directions], axis=0)

    scheme = acquisition_scheme_from_bvalues(bvalues, gradient_directions, delta, Delta)
    scheme.print_acquisition_info
    # ------INTRA AXONAL MODEL-------------
    # Cylinder orientation angles (theta, phi) == mu
    mu = np.array([0., 0.])
    # Parralel diffusivity lambda_par in E-9 m^2/s (in the paper d_par)
    lambda_par = 1.7e-9
    lambda_perp = 0.2e-9
    # Cylinder diameter in e-6 m (NOTE: alexander uses radi)
    diameter = 2 * 2.0e-6
    # Intra axonal tissue model using Van Gelderens signal model
    cylinder = cylinder_models.C4CylinderGaussianPhaseApproximation()

    # ----------EXTRA AXONAL MODEL-----------------
    zeppelin = gaussian_models.G2Zeppelin()

    mc_model = MultiCompartmentModel(models=[zeppelin, cylinder])

    # Fixing the parralel diffusivity parameter to be equal for intra and extra axonal models
    mc_model.set_equal_parameter('C4CylinderGaussianPhaseApproximation_1_lambda_par', 'G2Zeppelin_1_lambda_par')

    tissue_parameters = {
        'G2Zeppelin_1_mu': mu,
        'G2Zeppelin_1_lambda_par': lambda_par,
        'G2Zeppelin_1_lambda_perp': lambda_perp,
        'C4CylinderGaussianPhaseApproximation_1_mu': mu,
        'C4CylinderGaussianPhaseApproximation_1_lambda_par': lambda_par,
        'C4CylinderGaussianPhaseApproximation_1_diameter': diameter,
        'partial_volume_0': 0.5,
        'partial_volume_1': 0.5
    }

    signal = mc_model.simulate_signal(scheme, tissue_parameters)

    mc_model.set_initial_guess_parameter('C4CylinderGaussianPhaseApproximation_1_diameter', diameter)

    fitted_model = mc_model.fit(scheme, signal)

    print("\nGround truth parameter values: ", )
    pprint.pprint(tissue_parameters)
    print('\n')
    print("Fit result: ")
    pprint.pprint(fitted_model.fitted_parameters)


if __name__ == '__main__':
    main()

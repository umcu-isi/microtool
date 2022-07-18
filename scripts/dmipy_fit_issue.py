import pprint

import numpy as np
from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.signal_models import cylinder_models, gaussian_models

from microtool.utils import saved_schemes


def main():
    # Alexander uses 4 measurements for every gradient direction. The combinations of gradients and pulse timings per
    # measurement are the same for all directions (they are allowed to be different between measurements but not
    # (directions)

    # --------------- Acquisition Scheme ----------------

    scheme = saved_schemes.alexander2008()
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

    tissue_parameters_gt = {
        'G2Zeppelin_1_mu': mu,
        'G2Zeppelin_1_lambda_par': lambda_par,
        'G2Zeppelin_1_lambda_perp': lambda_perp,
        'C4CylinderGaussianPhaseApproximation_1_mu': mu,
        'C4CylinderGaussianPhaseApproximation_1_lambda_par': lambda_par,
        'C4CylinderGaussianPhaseApproximation_1_diameter': diameter,
        'partial_volume_0': 0.5,
        'partial_volume_1': 0.5
    }

    # --------- Initialization model --------------
    # We make a stick zeppelin to get initial values for the cylinder zeppelin model
    stick = cylinder_models.C1Stick()
    mc_model_init = MultiCompartmentModel(models=[zeppelin, stick])

    # Using the full model to simulate the signal
    signal = mc_model.simulate_signal(scheme, tissue_parameters_gt)
    # Using the stick model to estimate the parameters in the cylinder zeppelin model
    initial_fit = mc_model_init.fit(scheme, signal)
    parameter_guess = initial_fit.fitted_parameters

    tissue_parameters_init = {
        # We set the zeppelin values to the values found by stick zeppelin fitting
        'G2Zeppelin_1_mu': parameter_guess['G2Zeppelin_1_mu'],
        'G2Zeppelin_1_lambda_par': parameter_guess['G2Zeppelin_1_lambda_par'],
        'G2Zeppelin_1_lambda_perp': parameter_guess['G2Zeppelin_1_lambda_perp'],

        # For the cylinder we initialize the orientation and parralel diffusivities to those found by fitting stick zep
        'C4CylinderGaussianPhaseApproximation_1_mu': parameter_guess["C1Stick_1_mu"],
        'C4CylinderGaussianPhaseApproximation_1_lambda_par': parameter_guess["C1Stick_1_lambda_par"],

        # The diameter we initialize to the ground truth value as done by alexander.
        'C4CylinderGaussianPhaseApproximation_1_diameter': diameter,
        'partial_volume_0': parameter_guess['partial_volume_0'],
        'partial_volume_1': parameter_guess['partial_volume_1']
    }

    # Setting the initial values ("cascading" the parameters)
    for name, value in tissue_parameters_init.items():
        if name in mc_model.parameter_names:
            mc_model.set_initial_guess_parameter(name, value)

    fitted_model = mc_model.fit(scheme, signal)

    print("\nGround truth parameter values: ", )
    pprint.pprint(tissue_parameters_gt)
    print('\n')
    print("Fit result: ")
    pprint.pprint(fitted_model.fitted_parameters)


if __name__ == '__main__':
    main()

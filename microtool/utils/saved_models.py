import numpy as np
from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.signal_models import cylinder_models, gaussian_models, sphere_models

from microtool.dmipy import DmipyTissueModel


def verdict() -> DmipyTissueModel:
    # The three models we will be using to model intra cellular, extracellular and Vascular signal respectively
    # Initialize diameter of the cell i.e. sphere in the middle of optimization domain
    sphere = sphere_models.S4SphereGaussianPhaseApproximation(diffusion_constant=2.0e-9, diameter=10e-6)
    ball = gaussian_models.G1Ball(lambda_iso=2.0e-9)
    stick = cylinder_models.C1Stick(mu=[np.pi / 2, np.pi / 2], lambda_par=8.0e-9)
    verdict_model = MultiCompartmentModel(models=[sphere, ball, stick])
    # We fix the extracellular diffusivity in accordance with panagiotakis black magic number from 2014
    # verdict_model.set_fixed_parameter('G1Ball_1_lambda_iso', 2.0e-9)
    # verdict_model.set_fixed_parameter('C1Stick_1_lambda_par', 8.0e-9)
    # We set the optimization such that the pseudo diffusion coefficient is larger than 3.05 um^2/ms
    # verdict_model.set_parameter_optimization_bounds('C1Stick_1_lambda_par', [3.05e-9, 10e-9])

    verdict_model = DmipyTissueModel(verdict_model, [.3, .3, .4])
    # verdict_model['partial_volume_2'].optimize = False
    # verdict_model['partial_volume_1'].optimize = False
    # verdict_model['C1Stick_1_mu_0'].optimize = False
    # verdict_model['C1Stick_1_mu_1'].optimize = False
    return verdict_model


def cylinder_zeppelin_naked() -> MultiCompartmentModel:
    """
     A function to build the tissuemodel used in Alexander 2008.
    :return:
    """
    # ------INTRA AXONAL MODEL-------------
    # Cylinder orientation angles theta, phi := mu
    mu = np.array([np.pi / 2, 0.])
    # Parralel diffusivity lambda_par in E-9 m^2/s (in the paper d_par)
    lambda_par = 1.7e-9
    lambda_perp = 0.2e-9
    # Cylinder diameter in e-6 m (NOTE: alexander uses radi)
    diameter = 2 * 2.0e-6
    # Intra axonal tissue model using Van Gelderens signal model
    cylinder = cylinder_models.C4CylinderGaussianPhaseApproximation(mu, lambda_par, diameter, lambda_perp)

    # ----------EXTRA AXONAL MODEL-----------------
    zeppelin = gaussian_models.G2Zeppelin(mu, lambda_par, lambda_perp)

    mc_model = MultiCompartmentModel(models=[zeppelin, cylinder])

    # Fixing the parralel diffusivity parameter to be equal for intra and extra axonal models
    mc_model.set_equal_parameter('C4CylinderGaussianPhaseApproximation_1_lambda_par', 'G2Zeppelin_1_lambda_par')
    # Setting the initial diameter to the ground truth
    mc_model.set_initial_guess_parameter('C4CylinderGaussianPhaseApproximation_1_diameter', diameter)
    return mc_model


def cylinder_zeppelin() -> DmipyTissueModel:
    """
     A function to build the tissuemodel used in Alexander 2008.
    :return:
    """
    mc_model = cylinder_zeppelin_naked()

    # Wrapping the model for compatibility
    mc_model_wrapped = DmipyTissueModel(mc_model, volume_fractions=[.5, .5])

    return mc_model_wrapped


def stick_zeppelin() -> DmipyTissueModel:
    # Cylinder orientation angles theta, phi := mu
    mu = np.array([np.pi / 2, 0.])
    # Parralel diffusivity lambda_par in E-9 m^2/s (in the paper d_par)
    lambda_par = 1.7e-9
    lambda_perp = 0.2e-9

    zeppelin = gaussian_models.G2Zeppelin(mu, lambda_par, lambda_perp)
    stick = cylinder_models.C1Stick(mu, lambda_par)
    stick_zeppelin = MultiCompartmentModel(models=[zeppelin, stick])
    return DmipyTissueModel(stick_zeppelin, volume_fractions=[0.5, 0.5])


def stick() -> DmipyTissueModel:
    return DmipyTissueModel(stick_naked())


def stick_naked() -> MultiCompartmentModel:
    # Simplest model with orientation parameters
    mu = (np.pi / 2, np.pi / 2)  # in radians
    lambda_par = 1.7e-9  # in m^2/s
    return MultiCompartmentModel(models=[cylinder_models.C1Stick(mu=mu, lambda_par=lambda_par)])

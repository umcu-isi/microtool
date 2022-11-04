import numpy as np
from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.signal_models import cylinder_models, gaussian_models

from microtool.dmipy import DmipyTissueModel


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

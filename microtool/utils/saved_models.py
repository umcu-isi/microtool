from typing import List, Union

import numpy as np
from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.signal_models import cylinder_models, gaussian_models, sphere_models

from microtool.dmipy import DmipyTissueModel

Orientation = Union[List[float], np.ndarray]

#TODO: Review with new dmipy translation from dmipy branch

def verdict() -> DmipyTissueModel:
    """
    A function to build a VERDICT tissue model, as utilized by Panagiotaki et al [1]. The models hereby are utilized
    to model the Vascular, Extracellular and Restricted Diffusion for Cytometry in Tumors. 

    :return: a MICROTool compatible tissue model based on Dmipy's Cylinder-Gaussian phase, Ball and Stick compartment
    models.

    [1] Panagiotaki, E., Walker-Samuel, S., Siow, B., Johnson, S. P., Rajkumar, V., Pedley, R. B., ... & Alexander,
    D. C. (2014). Noninvasive quantification of solid tumor microstructure using VERDICT MRI. Cancer research, 74(7), 
    1902-1912.
    """
    # The three models we will be using to model intra cellular, extracellular and Vascular signal respectively
    # Initialize diameter of the cell i.e. sphere in the middle of optimization domain
    sphere = sphere_models.S4SphereGaussianPhaseApproximation(diffusion_constant=2.0e-9, diameter=10e-6)
    ball = gaussian_models.G1Ball(lambda_iso=2.0e-9)
    stick = cylinder_models.C1Stick(mu=[np.pi / 2, np.pi / 2], lambda_par=8.0e-9)
    verdict_model = MultiCompartmentModel(models=[sphere, ball, stick])

    verdict_model = DmipyTissueModel(verdict_model, [.3, .3, .4])

    return verdict_model


def cylinder_zeppelin_naked(orientation: Orientation) -> MultiCompartmentModel:
    """
    A function to build the tissuemodel used in Alexander 2008 [2]. As described, the model constitutes a 
    simplified version of Assaf et al.'s CHARMED model [3], with the addition of cylinder attenuation from that 
    of Van Gelderen's [4]. 

    :param orientation: The orientation of the cylinder zeppelin combination
    :return: a MICROTool compatible tissue model based on Dmipy's Zeppelin and Cylinder-Gaussian phase compartment 
    models.

    [2] Alexander, D. C. (2008). A general framework for experiment design in diffusion MRI and its application 
    in measuring direct tissue‐microstructure features. Magnetic Resonance in Medicine: An Official Journal of 
    the International Society for Magnetic Resonance in Medicine, 60(2), 439-448.
    [3] Assaf, Y., Freidlin, R. Z., Rohde, G. K., & Basser, P. J. (2004). New modeling and experimental framework
    to characterize hindered and restricted water diffusion in brain white matter. Magnetic Resonance in Medicine: 
    An Official Journal of the International Society for Magnetic Resonance in Medicine, 52(5), 965-978.
    [4] Vangelderen, P., DesPres, D., Vanzijl, P. C. M., & Moonen, C. T. W. (1994). Evaluation of restricted diffusion
    in cylinders. Phosphocreatine in rabbit leg muscle. Journal of Magnetic Resonance, Series B, 103(3), 255-260.
    """
    # ------INTRA AXONAL MODEL-------------
    # Cylinder orientation angles theta, phi := mu
    mu = np.array(orientation)
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

    # Wrapping the model for compatibility
    mc_model_wrapped = DmipyTissueModel(mc_model, volume_fractions=[.7, .3])

    return mc_model_wrapped
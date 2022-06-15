"""
Here we run the dmipy monte carlo simulation for the tissuemodel described in Alexander 2008:
https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.21646

"""
import pathlib
import pickle

import numpy as np
from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.signal_models import cylinder_models, gaussian_models
from scipy import stats

from microtool import monte_carlo
from microtool.dmipy import DmipyTissueModel
from microtool import utils_dmipy
from microtool.acquisition_scheme import DiffusionAcquisitionScheme

currentdir = pathlib.Path('.')
outputdir = currentdir / "MC_results"
outputdir.mkdir(exist_ok=True)


def main():
    # ------------- Setting up dmipy objects -----------
    noise_var = 0.02
    # -------------ACQUISITION-------------------
    # Define the PGSE acquisition scheme
    # First we sample the gradient directions
    n_pulses = 20
    angles = utils_dmipy.sample_sphere(n_pulses)  # for now random sampling from the sphere (not efficient or realistic)
    b_vectors = utils_dmipy.angles_to_eigenvectors(angles)
    b_values = np.repeat(10., n_pulses)
    pulse_widths = np.repeat(30., n_pulses)
    pulse_intervals = np.repeat(300., n_pulses)
    scheme = DiffusionAcquisitionScheme(b_values, b_vectors, pulse_widths, pulse_intervals)

    # ------INTRA AXONAL MODEL-------------
    # Cylinder orientation angles theta, phi := mu
    mu = np.array([0., 0.])
    # Parralel diffusivity lambda_par in E9 m^2/s (in the paper d_par)
    lambda_par = 1.7
    # some reason the dmipy docs claim this one is not scaled??
    lambda_perp = 0.2e-9
    # Cylinder diameter in m (NOTE: alexander uses radi)
    diameter = 2 * 2.0e-6
    # Intra axonal tissue model using Van Gelderens signal model
    cylinder = cylinder_models.C4CylinderGaussianPhaseApproximation(mu, lambda_par, diameter, lambda_perp)

    # ----------EXTRA AXONAL MODEL-----------------
    lambda_par *= 1.0e-9
    zeppelin = gaussian_models.G2Zeppelin(mu, lambda_par, lambda_perp)

    mc_model = MultiCompartmentModel(models=[cylinder, zeppelin])

    mc_model_wrapped = DmipyTissueModel(mc_model)

    # ----------- Optimizing the scheme ------------------
    mc_model_wrapped.optimize(scheme, noise_var)

    # ------------ Monte Carlo --------------------
    # Setting up the noise distribution

    noise_distribution = stats.norm(loc=0, scale=noise_var)

    # Running monte carlo simulation
    n_sim = 10

    tissue_parameters = monte_carlo.run(scheme, mc_model_wrapped, noise_distribution, n_sim)

    with open(outputdir / "TPD_alexander2008.pkl", "wb") as f:
        pickle.dump(tissue_parameters, f)


if __name__ == "__main__":
    main()

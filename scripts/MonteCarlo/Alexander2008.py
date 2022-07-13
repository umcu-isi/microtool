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
from microtool.utils import schemes
from microtool.dmipy import DmipyTissueModel

currentdir = pathlib.Path(__file__).parent
outputdir = currentdir / "results"
outputdir.mkdir(exist_ok=True)


def main():
    # ------------- Setting up dmipy objects -----------
    noise_var = 0.02
    # -------------ACQUISITION-------------------
    scheme = schemes.alexander2008()

    # ------INTRA AXONAL MODEL-------------
    # Cylinder orientation angles theta, phi := mu
    mu = np.array([np.pi/2, 0.])
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
    mc_model_wrapped = DmipyTissueModel(mc_model, volume_fractions=np.array([.5, .5]))

    print("Using the following model:\n", mc_model_wrapped)

    # ----------- Optimizing the scheme ------------------
    # mc_model_wrapped.optimize(scheme, noise_var)
    print("Using the optimized scheme:\n", scheme)

    # ------------ Monte Carlo --------------------
    # Setting up the noise distribution

    noise_distribution = stats.norm(loc=0, scale=noise_var)

    # Running monte carlo simulation
    n_sim = 1000

    tissue_parameters = monte_carlo.run(scheme, mc_model_wrapped, noise_distribution, n_sim, cascade=True)

    with open(outputdir / "alexander_nofixed_n_sim_{}_noise_{}.pkl".format(n_sim, noise_var), "wb") as f:
        pickle.dump(tissue_parameters, f)

    with open(outputdir / "alexander2008_ground_truth.pkl", 'wb') as f:
        pickle.dump(mc_model_wrapped, f)


if __name__ == "__main__":
    main()

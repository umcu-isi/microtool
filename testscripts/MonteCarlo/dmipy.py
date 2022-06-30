"""
Here we run the dmipy monte carlo simulation
"""
import pathlib
import pickle

import numpy as np
from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.data import saved_acquisition_schemes
from dmipy.signal_models import cylinder_models
from scipy import stats

from microtool import monte_carlo
from microtool.dmipy import DmipyTissueModel, DmipyAcquisitionSchemeWrapper

currentdir = pathlib.Path('..')
outputdir = currentdir / "results"
outputdir.mkdir(exist_ok=True)


def main():
    # ------------- Setting up dmipy objects -----------
    # predefined dmipy acquisition scheme
    acq_scheme = saved_acquisition_schemes.wu_minn_hcp_acquisition_scheme()
    scheme = DmipyAcquisitionSchemeWrapper(acq_scheme)
    # simplest tissuemodel available in dmipy
    mu = (np.pi / 2., np.pi / 2.)  # in radians
    lambda_par = 1.7e-9  # in m^2/s
    stick = cylinder_models.C1Stick(mu=mu, lambda_par=lambda_par)
    stick_model = MultiCompartmentModel(models=[stick])

    stick_model_wrapped = DmipyTissueModel(stick_model)
    # ------------ Monte Carlo --------------------
    # Setting up the noise distribution
    noise_var = 0.02
    noise_distribution = stats.norm(loc=0, scale=noise_var)

    # Running monte carlo simulation
    n_sim = 10

    stick_model_wrapped.optimize(scheme, noise_var)

    tissue_parameters = monte_carlo.run(scheme, stick_model_wrapped, noise_distribution, n_sim)

    with open(outputdir / "TPD.pkl", "wb") as f:
        pickle.dump(tissue_parameters, f)


if __name__ == "__main__":
    main()

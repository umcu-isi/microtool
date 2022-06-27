"""
This module contains all functions and classes involved in the monte carlo simulation of acquisition schemes
and how the noise distribution affects the estimated tissueparameters.

IMPORTANT: Always execute run inside a if name main clause! otherwise the parralel processing throws a cryptic error
"""
from typing import Union, List, Dict

import numpy as np
from scipy import stats
from scipy.stats.sampling import NumericalInversePolynomial

from tqdm import tqdm
from .acquisition_scheme import AcquisitionScheme
from .tissue_model import TissueModel
from .utils_IO import HiddenPrints

MonteCarloResult = List[Dict[str, Union[float, np.ndarray]]]


def run(scheme: AcquisitionScheme, model: TissueModel, noise_distribution: stats.rv_continuous,
        n_sim: int, test_mode = False) -> MonteCarloResult:
    """
    NEEDS TO BE EXECUTED IN if __name__ == "__main__" clause!!!! otherwise obscure parralel processing error.
    This function runs a Monte Carlo simulation to compute the posterior probability distribution induced in a
    tissue model given an aquisition scheme and a noise distribution.

    :param scheme: The acquisition scheme under investigation
    :param model: The tissuemodel for which we wish to know the posterior distribution
    :param noise_distribution: The noise distribution to perturb the signal
    :param n_sim: The number of times we sample the noise and fit the tissue parameters
    :return: The fitted tissueparameters for all noise realizations
    """

    # synthesize data from TissueModel with a set of tissueparameters using scheme
    signal = model(scheme)

    # Setting up the distribution sampler
    urng = np.random.default_rng()  # Numpy uniform distribution sampler as basis random number generator
    sampler = NumericalInversePolynomial(noise_distribution,
                                         random_state=urng)  # using scipy method to shape to distribution


    posterior = []
    for _ in tqdm(range(n_sim), desc=f"Starting Monte Carlo with {n_sim} simulations"):
        # get the noise level by sampling the given distribution for all individual measurements
        noise_level = sampler.rvs(size=len(signal))

        # add noise to the 'simulated' signal
        signal_bar = signal + noise_level

        # Fit the tissuemodel to the noisy data and save resulting parameters (hiding useless print statements
        with HiddenPrints():
            if test_mode:
                model_fitted = model.fit(scheme, signal)
            else:
                model_fitted = model.fit(scheme, signal_bar)
        parameter_dict = model_fitted.fitted_parameters

        # storing only information of interest namely the parameter values
        posterior.append(parameter_dict)

    return posterior

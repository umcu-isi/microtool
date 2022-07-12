"""
This module contains all functions and classes involved in the monte carlo simulation of acquisition schemes
and how the noise distribution affects the estimated tissueparameters.

IMPORTANT: Always execute run inside a if name main clause! otherwise the parralel processing throws a cryptic error
"""
from typing import Union, List, Dict

import numpy as np
from scipy import stats
from scipy.stats.sampling import NumericalInversePolynomial

from dmipy.signal_models import cylinder_models, gaussian_models
from dmipy.core.modeling_framework import MultiCompartmentModel
from tqdm import tqdm
from .acquisition_scheme import AcquisitionScheme
from .tissue_model import TissueModel
from microtool.utils.IO import HiddenPrints

MonteCarloResult = List[Dict[str, Union[float, np.ndarray]]]

INIT_MODEL = MultiCompartmentModel(models=[gaussian_models.G2Zeppelin(), cylinder_models.C1Stick()])


def run(scheme: AcquisitionScheme, model: TissueModel, noise_distribution: stats.rv_continuous,
        n_sim: int, cascade: bool = True, test_mode=False, **fit_options) -> MonteCarloResult:
    """
    NEEDS TO BE EXECUTED IN if __name__ == "__main__" clause!!!! otherwise obscure parralel processing error.
    This function runs a Monte Carlo simulation to compute the posterior probability distribution induced in a
    tissue model given an aquisition scheme and a noise distribution.


    :param scheme: The acquisition scheme under investigation
    :param model: The tissuemodel for which we wish to know the posterior distribution
    :param noise_distribution: The noise distribution to perturb the signal
    :param n_sim: The number of times we sample the noise and fit the tissue parameters
    :param test_mode: No noise added if true. This boolean can be used to test the fitting routine.
    :param cascade: Initial fit is done using stick_zeppelin model if true.
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
                model_fitted = _fit(model, scheme, signal, cascade, **fit_options)
            else:
                model_fitted = _fit(model, scheme, signal_bar, cascade, **fit_options)
        parameter_dict = model_fitted.fitted_parameters

        # storing only information of interest namely the parameter values
        posterior.append(parameter_dict)

    return posterior


def _fit(model: TissueModel, scheme, signal, cascade, **fitoptions):
    if cascade:
        guess = INIT_MODEL.fit(scheme, signal, **fitoptions)
        guess = _stickzeppelin_to_cylinderzeppelin(guess.fitted_parameters)
        model.set_initial_parameters(guess)
    return model.fit(scheme, signal)


def _stickzeppelin_to_cylinderzeppelin(parameters: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        # We set the zeppelin values to the values found by stick zeppelin fitting
        'G2Zeppelin_1_mu': parameters['G2Zeppelin_1_mu'],
        'G2Zeppelin_1_lambda_par': parameters['G2Zeppelin_1_lambda_par'],
        'G2Zeppelin_1_lambda_perp': parameters['G2Zeppelin_1_lambda_perp'],

        # For the cylinder we initialize the orientation and parralel diffusivities to those found by fitting stick zep
        'C4CylinderGaussianPhaseApproximation_1_mu': parameters["C1Stick_1_mu"],
        'C4CylinderGaussianPhaseApproximation_1_lambda_par': parameters["C1Stick_1_lambda_par"],

        'partial_volume_0': parameters['partial_volume_0'],
        'partial_volume_1': parameters['partial_volume_1']
    }

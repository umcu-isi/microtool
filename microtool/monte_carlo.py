"""
This module contains all functions and classes involved in the monte carlo simulation of acquisition schemes
and how the noise distribution affects the estimated tissueparameters.

IMPORTANT: Always execute run inside a if name main clause! otherwise the parralel processing throws a cryptic error
"""
from typing import Union, List, Dict

import numpy as np
from scipy import stats
from tqdm import tqdm

from microtool.utils.IO import HiddenPrints
from .acquisition_scheme import AcquisitionScheme
from .tissue_model import TissueModel

MonteCarloResult = List[Dict[str, Union[float, np.ndarray]]]


class MonteCarloSimulation:
    def __init__(self, scheme: AcquisitionScheme, model: TissueModel, noise_distribution: stats.rv_continuous):
        self._scheme = scheme
        self._model = model
        self.noise_distribution = noise_distribution

    def run(self, n_sim: int, noise_scale: float, test_mode: bool = True, **fit_options) -> MonteCarloResult:
        """

        :param n_sim:
        :param noise_scale:
        :param fit_options:
        :param test_mode:
        :return:
        """
        # Setting up the noise sampler
        sampler = self.noise_distribution(loc=0, scale=noise_scale)
        # Generating the unnoisy signal
        signal = self._model(self._scheme)

        posterior = []
        for _ in tqdm(range(n_sim), desc=f"Starting Monte Carlo with {n_sim} simulations"):
            # get the noise level by sampling the given distribution for all individual measurements
            noise_level = sampler.rvs(size=len(signal))

            # add noise to the 'simulated' signal
            signal_bar = signal + noise_level

            # Fit the tissuemodel to the noisy data and save resulting parameters (hiding useless print statements
            with HiddenPrints():
                if test_mode:
                    model_fitted = self._model.fit(self._scheme, signal)
                else:
                    model_fitted = self._model.fit(self._scheme, signal_bar)
            parameter_dict = model_fitted.fitted_parameters

            # storing only information of interest namely the parameter values
            posterior.append(parameter_dict)

        return posterior

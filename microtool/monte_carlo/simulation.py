"""
This module contains all functions and classes involved in the monte carlo simulation of acquisition schemes
and how the noise distribution affects the estimated tissueparameters.

IMPORTANT: Always execute run inside a if name main clause! otherwise the parralel processing throws a cryptic error
"""
import pathlib
import pickle

import pandas as pd
from scipy import stats
# scipy frozen distribution should be provided
from scipy.stats._distn_infrastructure import rv_continuous_frozen
from tqdm import tqdm

from ..acquisition_scheme import AcquisitionScheme
from ..tissue_model import TissueModel
from ..utils.IO import HiddenPrints


class MonteCarloSimulation:
    _fit_options = {}
    _result = None

    def __init__(self, scheme: AcquisitionScheme, model: TissueModel, noise_distribution: rv_continuous_frozen,
                 n_sim: int):
        """
        Handles the running and saving of a monte carlo simulation. Has setters for relevant parameters such that object
        can be reused for simulation with a different control parameter.

        :param scheme: The AcquisitionScheme with which we generate signal
        :param model: The TissueModel with which we generate signal AND that is used for fitting
        :param noise_distribution: The noise distribution from which we sample noise for every measurement
        :param n_sim: Number of fit repetitions
        """
        self._scheme = scheme
        self._model = model
        self._n_sim = n_sim
        self._noise_distribution = noise_distribution

    def __str__(self) -> str:
        firstline = "Initialized MonteCarloSimulation with scheme for signal generation:\n"
        schemestr = self._scheme.__str__() + "\n"
        secondline = "Fitting tissuemodel: \n "
        modelstr = self._model.__str__() + "\n"

        distname = self._noise_distribution.dist.name
        pars = self._noise_distribution.kwds
        thirdline = f"Generating noise according to a {distname} distribution with parameters {pars}"
        return firstline + schemestr + secondline + modelstr + thirdline

    def set_noise_distribution(self, noise_distribution: stats.rv_continuous) -> None:
        self._noise_distribution = noise_distribution

    def set_fitting_options(self, fit_options: dict) -> None:
        self._fit_options = fit_options

    def set_scheme(self, scheme: AcquisitionScheme) -> None:
        self._scheme = scheme

    def set_model(self, model: TissueModel) -> None:
        self._model = model

    def set_n_sim(self, n_sim: int) -> None:
        self._n_sim = n_sim

    def run(self) -> pd.DataFrame:
        """

        :return:
        """
        n_sim = self._n_sim
        # Setting up the noise sampler
        sampler = self._noise_distribution

        # Generating the unnoisy signal
        signal = self._model(self._scheme)

        result = []
        for _ in tqdm(range(n_sim), desc=f"Starting Monte Carlo with {n_sim} simulations"):
            # get the noise level by sampling the given distribution for all individual measurements
            noise_level = sampler.rvs(size=len(signal))

            # add noise to the 'simulated' signal
            signal_bar = signal + noise_level

            # Fit the tissuemodel to the noisy data and save resulting parameters (hiding useless print statements
            with HiddenPrints():
                model_fitted = self._model.fit(self._scheme, signal_bar, **self._fit_options)

            parameter_dict = model_fitted.fitted_parameters
            # storing only information of interest namely the parameter values
            result.append(parameter_dict)
        self._result = pd.DataFrame(result)
        return self._result

    def save(self, datadir: pathlib.Path, model_name: str, scheme_name: str) -> None:
        """
        Saves the montecarlo result and the scheme and model used during the simulation.
        :param datadir: A path with folder and name included
        """
        if self._result is None:
            raise RuntimeError("You cannot save a simulation that has not been run yet.")

        sim_name = self._save_name

        # save ground truth
        with open(datadir / (model_name + "_gt.pkl"), 'wb') as f:
            pickle.dump(self._model, f)

        # save scheme
        with open(datadir / (scheme_name + "_scheme.pkl"), 'wb') as f:
            pickle.dump(self._scheme, f)

        # save simulation result
        with open(datadir / (model_name + scheme_name + sim_name + ".pkl"), 'wb') as f:
            pickle.dump(self._result, f)

    @property
    def _save_name(self) -> str:
        scale = self._noise_distribution.kwds["scale"]
        std = "{:.2f}".format(scale)
        return f"_nsim={self._n_sim}_{self._noise_distribution.dist.name}_sigma=" + std

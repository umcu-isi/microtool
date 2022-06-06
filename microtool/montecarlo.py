"""
This module contains all functions and classes involved in the monte carlo simulation of acquisition schemes
and how the noise distribution affects the estimated tissueparameters.
"""
from typing import List, Dict, Any

from tissue_model import TissueModel
from acquisition_scheme import AcquisitionScheme
import numpy as np


def monte_carlo_run(scheme: AcquisitionScheme, model: TissueModel, noise_distribution: callable, n_sim: int) -> list[
    Dict]:
    # synthesize data from TissueModel with a set of tissueparameters using scheme
    signal = model(scheme)

    posterior = []
    for i in range(n_sim):
        # get the noise level by sampling the given distribution
        noise_level = sample(noise_distribution)

        # add noise to the 'simulated' signal
        signal_bar = signal + noise_level

        # Fit the tissuemodel to the noisy data and save resulting parameters
        posterior.append(model.fit(scheme, signal_bar))

    return posterior


def sample(noise_distribution: callable) -> float:
    return 0.0

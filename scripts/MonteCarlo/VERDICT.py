import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from microtool.monte_carlo import MonteCarloSimulation
from microtool.monte_carlo.IO import make_result_dirs
from microtool.monte_carlo.parameter_distributions import plot_parameter_distributions
from microtool.utils.IO import get_pickle

currentdir = pathlib.Path(__file__).parent
resultdir = currentdir / "results"
plotdir, datadir = make_result_dirs(resultdir, "VERDICT")

if __name__ == "__main__":
    # ---- load model
    verdict = get_pickle('../optimize/models/verdict_gt.pkl')
    # ---- load schemes
    optimal_scheme = get_pickle("../optimize/schemes/verdict_optimal.pkl")
    initial_scheme = get_pickle("../optimize/schemes/verdict_start.pkl")

    # ---- Monte carlo simulations
    # noise setup
    noise_variance = 0.002
    noise_distribution = stats.norm(loc=0, scale=np.sqrt(noise_variance))

    # optimal simulation
    simulation = MonteCarloSimulation(optimal_scheme, verdict, noise_distribution, n_sim=1000)
    simulation.set_fitting_options({"use_parallel_processing": False})
    optimal_result = simulation.run()
    simulation.save(datadir, "verdict", "optimal")

    # non -optimal simulation
    simulation.set_scheme(initial_scheme)
    non_optimal_result = simulation.run()
    simulation.save(datadir, "verdict", "non-optimal")

    # ---- plotting
    plot_parameter_distributions(optimal_result, verdict, fig_label="optimal")
    plot_parameter_distributions(non_optimal_result, verdict, fig_label="non_optimal")
    plt.show()

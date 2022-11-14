import pathlib

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from microtool.monte_carlo import MonteCarloSimulation, make_result_dirs
from microtool.monte_carlo.parameter_distributions import scale, plot_parameter_distributions
from microtool.tissue_model import ExponentialTissueModel
from microtool.utils import IO
from microtool.utils.plotting import plot_acquisition_parameters

currentdir = pathlib.Path(__file__).parent
resultdir = currentdir / "results"
plotdir, datadir = make_result_dirs(resultdir, "exponential_model")

if __name__ == "__main__":
    # set the noise
    noise_variance = 0.02
    noise_distribution = stats.norm(loc=0, scale=np.sqrt(noise_variance))

    model = ExponentialTissueModel(T2=10)
    scheme_start = IO.get_pickle("../optimize/schemes/exponential_start.pkl")
    scheme_optimal = IO.get_pickle("../optimize/schemes/exponential_optimal.pkl")

    # Monte Carlo simulations
    n_sim = 1000

    # running simulation with optimal scheme
    simulation = MonteCarloSimulation(scheme_optimal, model, noise_distribution, n_sim)
    result = simulation.run()
    simulation.save(datadir, "T2=10", "optimized")

    # doing same for non optimal scheme
    simulation.set_scheme(scheme_start)
    non_optimal_result = simulation.run()
    simulation.save(datadir, "T2=10", "non-optimized")

    # ------------ plotting

    # scaling the distributions with respect to the groundtruth values in the tissuemodel
    optimal_scaled_parameters = scale(result, model)
    non_optimal_scaled_parameters = scale(non_optimal_result, model)

    # plotting the distributions
    symbols = {"T2": r"$T_2$ [ms]"}
    plot_parameter_distributions(result, model, symbols=symbols, fig_label="optimal")
    plt.savefig(plotdir / ("optimal_PD" + simulation._save_name + ".png"))
    plot_parameter_distributions(non_optimal_result, model, symbols=symbols, fig_label="non_optimal")
    plt.savefig(plotdir / ("non_optimal_PD" + simulation._save_name + ".png"))

    # plotting the aquisition parameters
    plot_acquisition_parameters(scheme_optimal)
    plt.savefig((plotdir / ("optimal_AP.png")))
    plot_acquisition_parameters(scheme_start)
    plt.savefig((plotdir / ("non_optimal_AP.png")))

    # plotting the signal

    plt.show()

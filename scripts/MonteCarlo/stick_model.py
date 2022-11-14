"""
Here we run the dmipy monte carlo simulation
"""
import pathlib

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from microtool.monte_carlo import MonteCarloSimulation, make_result_dirs
from microtool.monte_carlo.parameter_distributions import plot_parameter_distributions
from microtool.utils import saved_models, IO
from microtool.utils.plotting import plot_acquisition_parameters

currentdir = pathlib.Path(__file__).parent
resultdir = currentdir / "results"
plotdir, datadir = make_result_dirs(resultdir, "stick_model")


def main():
    # ------------- Setting up dmipy objects -----------
    # pre-optimized schemes
    scheme_opt = IO.get_pickle("../optimize/schemes/stick_scheme_optimal.pkl")
    scheme_start = IO.get_pickle("../optimize/schemes/stick_scheme_start.pkl")

    # predefined microtool-wrapped dmipy stickmodel
    stick_model = saved_models.stick()

    # ---------------- Setting up the noise distribution
    noise_var = 0.02
    noise_distribution = stats.norm(loc=0, scale=np.sqrt(noise_var))

    # ------------ Monte Carlo --------------------
    n_sim = 1000
    simulation = MonteCarloSimulation(scheme_opt, stick_model, noise_distribution, n_sim)
    simulation.set_fitting_options({"use_parallel_processing": False})

    # for optimal scheme
    result = simulation.run()
    simulation.save(datadir, "stick_model", "optimized")

    # for non optimal scheme
    simulation.set_scheme(scheme_start)
    result_start = simulation.run()
    simulation.save(datadir, model_name="stick", scheme_name='start')

    plot_parameter_distributions(result, stick_model, fig_label="optimal")
    plt.suptitle("optimal")
    plt.tight_layout()
    plot_acquisition_parameters(scheme_opt)

    plot_parameter_distributions(result_start, stick_model, fig_label="start")
    plt.suptitle("start")
    plt.tight_layout()
    plot_acquisition_parameters(scheme_start)

    plt.show()


if __name__ == "__main__":
    main()

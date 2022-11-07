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
    # predefined scheme
    scheme_opt = IO.get_pickle("../optimize/schemes/stick_scheme_optimal.pkl")
    # predefined microtool-wrapped dmipy stickmodel
    stick_model = saved_models.stick()

    # ---------------- Setting up the noise distribution
    noise_var = 0.02
    noise_distribution = stats.norm(loc=0, scale=np.sqrt(noise_var))

    # ------------ Monte Carlo --------------------
    n_sim = 2
    simulation = MonteCarloSimulation(scheme_opt, stick_model, noise_distribution, n_sim)
    result = simulation.run()
    simulation.save(datadir, "stick_model", "optimized")

    plot_parameter_distributions(result, stick_model)
    plot_acquisition_parameters(scheme_opt)
    plt.show()


if __name__ == "__main__":
    main()

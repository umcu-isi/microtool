import pathlib
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

from microtool.acquisition_scheme import EchoScheme
from microtool.monte_carlo import MonteCarloSimulation
from microtool.optimize import optimize_scheme
from microtool.tissue_model import ExponentialTissueModel
from microtool.utils.parameter_distributions import scale
from microtool.utils.plotting import plot_acquisition_parameters, plot_parameter_distributions

currentdir = pathlib.Path(__file__).parent
outputdir = currentdir / "results" / "exponential_model"
outputdir.mkdir(exist_ok=True)

if __name__ == "__main__":
    # set the noise
    noise = 0.02
    noise_distribution = stats.norm(loc=0, scale=np.sqrt(noise))

    # Aquisition scheme
    TE = np.linspace(5, 100, num=30)
    scheme = EchoScheme(TE)

    # Tissuemodel
    model = ExponentialTissueModel(T2=10)

    # saving the ground truth for plotting later
    with open(outputdir / "exponential_model_gt.pkl", 'wb') as f:
        pickle.dump(model, f)

    # optimization
    scheme_opt, _ = optimize_scheme(scheme, model, noise)

    # Monte Carlo simulations
    n_sim = 100

    # running simulation with optimal scheme
    simulation = MonteCarloSimulation(scheme_opt, model, noise_distribution, n_sim)
    print(simulation)
    result = simulation.run()
    simulation.save(outputdir / f"optimal_scheme_{n_sim}_gaussian")

    # doing same for non optimal scheme
    simulation.set_scheme(scheme)
    non_optimal_result = simulation.run()
    simulation.save(outputdir / f"non_optimal_scheme_{n_sim}_gaussian")

    # ------------ plotting

    # scaling the distributions with respect to the groundtruth values in the tissuemodel
    optimal_scaled_parameters = scale(pd.DataFrame(result), model)
    non_optimal_scaled_parameters = scale(pd.DataFrame(non_optimal_result), model)

    # plotting the distributions
    plot_parameter_distributions(optimal_scaled_parameters, fig_label="optimal")
    plot_parameter_distributions(non_optimal_scaled_parameters, fig_label="non_optimal")

    # plotting the aquisition parameters
    plot_acquisition_parameters(scheme_opt)
    plot_acquisition_parameters(scheme)
    plt.show()

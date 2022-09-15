import pathlib
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

from microtool import monte_carlo
from microtool.acquisition_scheme import EchoScheme
from microtool.optimize import optimize_scheme
from microtool.tissue_model import ExponentialTissueModel
from microtool.utils.parameter_distributions import scale
from microtool.utils.plotting import plot_acquisition_parameters, plot_parameter_distributions

currentdir = pathlib.Path(__file__).parent
outputdir = currentdir / "results" / "exponential_model_validation"
outputdir.mkdir(exist_ok=True)

if __name__ == "__main__":
    # set the noise
    noise = 0.02

    # Aquisition scheme
    TE = np.array([1, 30, 60])
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
    noise_distribution = stats.norm(loc=0, scale=noise)
    result = monte_carlo.run(scheme_opt, model, noise_distribution, n_sim=n_sim, cascade=False)

    # saving result
    with open(outputdir / "T2_distribution_optimal_scheme_nsim_{}.pkl".format(n_sim), 'wb') as f:
        pickle.dump(result, f)

    non_optimal_result = monte_carlo.run(scheme, model, noise_distribution, n_sim=n_sim, cascade=False)

    with open(outputdir / "T2_distribution_non_optimal_scheme_nsim_{}.pkl".format(n_sim), 'wb') as f:
        pickle.dump(non_optimal_result, f)


    # scaling the distributions with respect to the groundtruth values in the tissuemodel
    optimal_scaled_parameters = scale(pd.DataFrame(result), model)
    non_optimal_scaled_parameters = scale(pd.DataFrame(non_optimal_result),model)

    # plotting the distributions
    plot_parameter_distributions(optimal_scaled_parameters,fig_label="optimal")
    plot_parameter_distributions(non_optimal_scaled_parameters,fig_label="non_optimal")

    # plotting the aquisition parameters
    plot_acquisition_parameters(scheme_opt)
    plot_acquisition_parameters(scheme)
    plt.show()

import pathlib
import pprint

import pandas as pd
from matplotlib import pyplot as plt

from microtool.monte_carlo.parameter_distributions import plot_parameter_distributions
from microtool.utils.IO import get_pickle

resultdir = pathlib.Path(__file__).parent / 'results' / "exponential_model"

# the result with optimized scheme
filename = "T2_distribution_optimal_scheme_nsim_10000.pkl"
gt = get_pickle(resultdir / "exponential_model_gt.pkl")

df_optimal = pd.DataFrame(get_pickle(resultdir / filename))
df_non_optimal = pd.DataFrame(get_pickle(resultdir / "T2_distribution_non_optimal_scheme_nsim_10000.pkl"))

# plotting the distributions
plot_parameter_distributions(df_optimal, gt, fig_label='Optimal Echo Scheme')
plot_parameter_distributions(df_non_optimal, gt, fig_label="Non optimal echo scheme")

# Adding a table for the ground truth values
gt_dict = gt.parameters
pprint.pprint(gt_dict)

plt.show()

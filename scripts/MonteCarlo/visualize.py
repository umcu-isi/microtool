import pathlib
import pprint

import pandas as pd
from matplotlib import pyplot as plt

from microtool import monte_carlo
from microtool.utils.IO import get_pickle

resultdir = pathlib.Path(__file__).parent / 'results' / "exponential_model_validation"

# the result with optimized scheme
filename = "T2_distribution_optimal_scheme_nsim_10000.pkl"
df_optimal = pd.DataFrame(get_pickle(resultdir / filename))
gt = get_pickle(resultdir / "exponential_model_gt.pkl")

df_non_optimal = pd.DataFrame(get_pickle(resultdir / "T2_distribution_non_optimal_scheme_nsim_10000.pkl"))
monte_carlo.show(df_optimal, gt, 'Optimal Echo Scheme')
monte_carlo.show(df_non_optimal, gt, "Non optimal echo scheme")

# Adding a table for the ground truth values
gt_dict = gt.parameters
pprint.pprint(gt_dict)

plt.show()

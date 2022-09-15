import math
import pathlib

import matplotlib.colors as mcolors
import pandas as pd
from matplotlib import pyplot as plt

from microtool.utils.IO import get_df_from_pickle, get_pickle
from microtool.utils.plotting import plot_gaussian_fit, plot_dataframe_index
from microtool.utils.parameter_distributions import fit_gaussian, scale

resultdir = pathlib.Path(__file__).parent / 'results' / "shell_distribution_experiments"

gt = get_pickle(resultdir / "alexander2008_ground_truth.pkl")

files = ["alexander_shells_[8, 32, 80]_n_sim_100_noise_0.02.pkl",
         "alexander_shells_[20, 20, 80]_n_sim_100_noise_0.02.pkl",
         "alexander_shells_[30, 30, 60]_n_sim_100_noise_0.02.pkl",
         "alexander_shells_[40, 40, 40]_n_sim_100_noise_0.02.pkl"]


def main():
    colors = list(mcolors.TABLEAU_COLORS)
    fit_results_mean = {}
    fit_results_std = {}
    for i, filename in enumerate(files):
        # reading the parameter distrubtion from pickle file
        df = get_df_from_pickle(resultdir/filename)
        # Scaling w.r.t. the ground truth values from the tissuemodel
        df_scaled = scale(df, gt)
        fit_results = fit_gaussian(df_scaled)
        plot_gaussian_fit(df_scaled, fit_results, colors[i])
        fit_results_mean[filename.split('_')[2]] = fit_results['mean']
        fit_results_std[filename.split('_')[2]] = fit_results['std']

    # making a legend in the last subplot
    plt.legend()

    fit_results_std = pd.DataFrame.from_dict(fit_results_std, orient='columns')

    # making a bar chart where the mean value is represented for every parameter on the x-axis and different colors
    # for shell configurations
    n_rows = math.ceil(len(fit_results_std.index) / 3)
    fig, axes = plt.subplots(n_rows, 3)
    for i, parameter in enumerate(fit_results_std.index):
        plot_dataframe_index(fit_results_std, parameter, axes.flatten()[i])
    plt.suptitle("Standard deviations for different shell configurations")
    fit_results_std.plot.bar(rot=45, title='standard deviations of tissueparameters')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

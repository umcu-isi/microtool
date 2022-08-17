import math
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import norm
import pprint

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

from microtool.utils.IO import get_df_from_pickle, get_pickle

resultdir = pathlib.Path(__file__).parent / 'results'

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
        all_results = visualize_pickle(filename, colors[i])
        fit_results_mean[filename.split('_')[2]] = all_results['mean']
        fit_results_std[filename.split('_')[2]] = all_results['std']

    # making a legend in the last subplot
    plt.legend()

    fit_results_std = pd.DataFrame.from_dict(fit_results_std, orient='columns')
    fit_results_mean = pd.DataFrame.from_dict(fit_results_mean, orient='columns')

    # making a bar chart where the mean value is represented for every parameter on the x-axis and different colors
    # for shell configurations
    n_rows = math.ceil(len(fit_results_std.index) / 3)
    fig, axes = plt.subplots(n_rows, 3)
    j = 0
    for i, parameter in enumerate(fit_results_std.index):
        plot_df_index(fit_results_std, parameter, axes.flatten()[i])
    plt.suptitle("Standard deviations for different shell configurations")
    fit_results_std.plot.bar(rot=45, title='standard deviations of tissueparameters')
    plt.tight_layout()
    plt.show()


def plot_df_index(df: pd.DataFrame, index_name: str, ax):
    df.loc[index_name].to_frame(index_name).T.plot.bar(ylabel=r'std_fitted', xticks=[], title=index_name, ax=ax)


def visualize_pickle(filename: str, color: str = None) -> pd.DataFrame:
    df = get_df_from_pickle(resultdir / filename)
    n_rows = math.ceil(df.shape[1] / 3) + 1

    # print(df.describe())
    fit_results = {}
    for i, parameter in enumerate(df.keys()):
        ax = plt.subplot(n_rows, 3, i + 1)

        gt_parameter = gt[parameter]
        scale = gt_parameter.scale
        value = gt_parameter.value / gt_parameter.scale

        # computing scaled parameter values
        scaled_parameter_samples = df[parameter] / scale - value
        # Fitting a normal distribution to the samples
        fitted_mean, fitted_std = norm.fit(scaled_parameter_samples)
        fit_results[parameter] = {}
        fit_results[parameter]['mean'] = fitted_mean
        fit_results[parameter]['std'] = fitted_std
        # Making a histogram
        ax.hist(scaled_parameter_samples, bins='scott', alpha=0.5, label=filename.split('_')[2], color=color)
        # Plotting the fitted normal distribution as well
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        ax.plot(x, norm.pdf(x, fitted_mean, fitted_std), color=color)

        ax.set_xlabel(r"$\Delta$")
        # plotting ground truth as vertical lines
        ax.vlines(0, 0, 1, transform=ax.get_xaxis_transform(), colors="black")
        ax.set_title(parameter)

    # Adding a table for the ground truth values
    gt_dict = gt.parameters
    plt.tight_layout()

    # Outputting the fit results
    return pd.DataFrame.from_dict(fit_results, orient='index')


if __name__ == "__main__":
    main()

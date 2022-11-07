import math
from typing import Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm

from microtool.tissue_model import TissueModel


def fit_gaussian(parameter_distribution: pd.DataFrame) -> pd.DataFrame:
    """
    Function that returns a nested dataframe containing the results of gaussian fit.
    :param parameter_distribution:
    :return: Nested dataframe containing df[parameter]['mean'] and df[parameter]['std']
    """

    fit_results = {}
    for parameter in parameter_distribution.keys():
        fitted_mean, fitted_std = norm.fit(parameter_distribution[parameter])
        fit_results[parameter] = {}
        fit_results[parameter]['mean'] = fitted_mean
        fit_results[parameter]['std'] = fitted_std

    return pd.DataFrame.from_dict(fit_results, orient='index')


def scale(mc_result: pd.DataFrame, ground_truth: TissueModel) -> pd.DataFrame:
    """
    Helper function for rescaling the parameter distribution

    :param mc_result: Parameter distribution which we wish to rescale
    :param ground_truth: The ground truth tissue model w.r.t. we wish to scale
    :return: The scaled distribution
    """
    scaled_distribution = {}
    for parameter in mc_result.keys():
        gt_parameter = ground_truth[parameter]
        scale = gt_parameter.scale
        value = gt_parameter.value / gt_parameter.scale
        scaled_distribution[parameter] = mc_result[parameter] / scale - value

    return pd.DataFrame(scaled_distribution)


def plot_parameter_distributions(mc_result: pd.DataFrame, gt_model: TissueModel,
                                 symbols: Dict[str, str] = None,
                                 gaussian_fit: pd.DataFrame = None,
                                 color: str = None, fig_label: str = "parameter distributions") -> plt.Figure:
    """

    :param mc_result: The result from a montecarlo simulation
    :param gt_model: The groundtruth tissuemodel used in the montecarlo simulation
    :param symbols: A dict mapping a tissuemodel name to another string used to format the plot
    :param gaussian_fit: Result from fitting gaussian distributions to monte carlo result
    :param color: Color used in plotting
    :param fig_label:
    :return:
    """
    fig = plt.figure(fig_label)

    # Determining subplot layout based on number of parameters present in the distribution
    n_tissue_parameters = min(mc_result.shape)

    if n_tissue_parameters < 3:
        ncols = n_tissue_parameters
    else:
        ncols = 3

    n_rows = math.ceil(mc_result.shape[1] / ncols)

    for i, parameter in enumerate(mc_result.keys()):
        ax = plt.subplot(n_rows, ncols, i + 1)

        # Making a histogram
        ax.hist(mc_result[parameter], bins='scott', alpha=0.5, color=color)

        # Plotting the fitted normal distribution if provided
        if gaussian_fit is not None:
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            fitted_mean = gaussian_fit['mean'][parameter]
            fitted_std = gaussian_fit['std'][parameter]
            ax.plot(x, norm.pdf(x, fitted_mean, fitted_std), color=color, label="Fitted gaussian")

        # formatting the x-axis if symbols are provided
        if symbols is not None and (parameter in symbols.keys()):
            ax.set_xlabel(symbols[parameter])
        else:
            ax.set_xlabel(parameter)

        # plotting ground truth as vertical lines
        ax.vlines(gt_model[parameter].value, 0, 1, transform=ax.get_xaxis_transform(), colors="black",
                  label="Ground Truth")
        ax.set_title(parameter)
        plt.legend()

    plt.tight_layout()
    return fig


def plot_dataframe_index(df: pd.DataFrame, index_name: str, ax: plt.Axes) -> None:
    """
    Plots an index from a pandas dataframe to the given axes object
    :param df: The dataframe
    :param index_name: The index you wish to plot
    :param ax: The axes you wish to plot to
    :return:
    """
    df.loc[index_name].to_frame(index_name).T.plot.bar(ylabel=r'std_fitted', xticks=[], title=index_name, ax=ax)

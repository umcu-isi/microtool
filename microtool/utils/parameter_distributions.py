import pandas as pd
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


def scale(parameter_distribution: pd.DataFrame, ground_truth: TissueModel) -> pd.DataFrame:
    """
    Helper function for rescaling the parameter distribution

    :param parameter_distribution: Parameter distribution which we wish to rescale
    :param ground_truth: The ground truth tissue model w.r.t. we wish to scale
    :return: The scaled distribution
    """
    scaled_distribution = {}
    for parameter in parameter_distribution.keys():
        gt_parameter = ground_truth[parameter]
        scale = gt_parameter.scale
        value = gt_parameter.value / gt_parameter.scale
        scaled_distribution[parameter] = parameter_distribution[parameter] / scale - value

    return pd.DataFrame(scaled_distribution)

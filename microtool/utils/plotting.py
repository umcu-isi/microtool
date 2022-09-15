import math
from typing import Dict, List, Tuple

import pandas as pd
from scipy.stats import norm

from microtool.optimize import LossFunction
from microtool.acquisition_scheme import AcquisitionScheme
from microtool.tissue_model import TissueModel
from copy import copy
import numpy as np
import matplotlib.pyplot as plt

Domains = List[Tuple[float, float]]


class LossInspector:
    def __init__(self, loss_function: LossFunction, scheme: AcquisitionScheme, model: TissueModel, noise_var: float):
        """

        :param loss_function: The loss function we wish to inspect
        :param scheme: The (optimized) acquisition scheme for which we wish to inspect the loss
        :param model: The tissuemodel under investigation
        :param noise_var: The noise variance estimation
        """
        self.scheme = copy(scheme)
        self.model = model
        self.noise_var = noise_var
        self.loss_function = loss_function

    def compute_loss(self, x: np.ndarray) -> float:
        """
        :param x: The scaled parameters for which we wish to know the loss
        :return: the loss value
        """
        tissue_scales = self.model.scales
        tissue_include = self.model.include
        acq_scales = self.scheme.free_parameter_scales
        # we update the scheme to compute the loss altough this does not change the scheme the user supplied
        self.scheme.set_free_parameter_vector(x * acq_scales)
        jac = self.model.jacobian(self.scheme)
        return self.loss_function(jac, tissue_scales, tissue_include, self.noise_var)

    def plot(self, parameters: Dict[str, int], domains: Domains = None) -> None:
        """
        Allows for plotting a loss function as a function of 2 or 1 parameter(s) close to a minimum found by
        optimization.

        :param parameters: A dictionary containing a free parameter name as key and the pulse for which you wish to
                           inspect the loss.
        :param domains: The domains for which you wish to inspect the parameters
        :raises: ValueError if domains are out of bounds or if arguments are incompatible
        """
        # check if the provided parameters are actually in the scheme
        for key in parameters.keys():
            if key not in self.scheme.free_parameters:
                raise ValueError("The provided acquisition parameter key(s) do not match the provided schemes free "
                                 f"parameters choices are {self.scheme.free_parameter_keys}")

        # check if provided pulse id is the correct value
        for key, pulse_id in parameters.items():
            if pulse_id >= self.scheme.pulse_count or pulse_id < 0:
                raise ValueError(f"Invalid pulse id provided for {key}.")

        x_optimal = self.scheme.free_parameter_vector / self.scheme.free_parameter_scales
        parameter_idx = [self.scheme.get_free_parameter_idx(key, value) for key, value in parameters.items()]
        parameters_optimal = [x_optimal[i] for i in parameter_idx]
        # make domains if not provided
        if domains:
            self._check_domains(parameters, domains)
            domains = np.array(domains)
        else:
            domains = self._make_domains(parameters)
            domains = np.array(domains)

        # discretizing domains
        domains = np.linspace(domains[:, 0], domains[:, 1], endpoint=True)

        # getting plotting parameters scales to get correct axes later
        scales = []
        for key in parameters.keys():
            scales.append(self.scheme[key].scale)

        # 3d plots for 2 investigated parameters
        if len(parameters) == 2:
            def loss(x1, x2):
                # return loss given two parameters we are changing and other parameters left constant
                params = x_optimal
                params[parameter_idx[0]] = x1
                params[parameter_idx[1]] = x2
                return self.compute_loss(params)

            # Vectorization so we can pass the discretized domains and compute on all values
            vloss = np.vectorize(loss)
            # make a meshgrid out of the two changing parameters, so we can compute loss on the entire grid
            X1, X2 = np.meshgrid(domains[:, 0], domains[:, 1])

            # making the figure
            fig = plt.figure("3D loss landscape")
            ax = plt.axes(projection='3d')
            Z = vloss(X1, X2)

            ax.plot_surface(X1 * scales[0], X2 * scales[1], Z, alpha=0.5, color='grey')
            ax.plot(*[parameters_optimal[i] * scales[i] for i in range(2)], loss(*parameters_optimal), 'ro',
                    label="Optimal point")
            ax.legend()
            ax.set_zlabel("Loss")
            labels = list(parameters.keys())
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            fig.tight_layout()
        else:
            # Normal plot if investigating 1 parameter
            def loss(x1):
                # return loss given x1
                params = x_optimal
                params[parameter_idx[0]] = x1
                return self.compute_loss(params)

            # vectorization if so we can pass the domain
            vloss = np.vectorize(loss)

            # making the figure
            plt.figure()
            plt.plot(domains[:, 0] * scales[0], vloss(domains[:, 0]))
            plt.plot(parameters_optimal[0] * scales[0], loss(parameters_optimal[0]), 'ro', label="Optimal point")
            plt.xlabel(list(parameters.keys())[0])
            plt.ylabel("Loss function")
            plt.legend()
            plt.tight_layout()

    def _make_domains(self, parameters: Dict[str, int]) -> Domains:
        """
        :param parameters: Dictionary with parameter key and pulse id
        :return: Domains of 0.1 around optimum (or at parameter boundary otherwise)
        """
        domains = []
        for parameter, pulse_id in parameters.items():
            # for default domains we just
            tissue_parameter = self.scheme[parameter]

            lb = tissue_parameter.values[pulse_id] / tissue_parameter.scale - 0.1
            ub = tissue_parameter.values[pulse_id] / tissue_parameter.scale + 0.1
            if tissue_parameter.upper_bound and ub > tissue_parameter.upper_bound:
                ub = tissue_parameter.upper_bound
            if tissue_parameter.lower_bound and lb < tissue_parameter.lower_bound:
                lb = tissue_parameter.lower_bound

            domains.append((lb, ub))
        return domains

    def _check_domains(self, parameters, domains):
        """
        :param parameters: Dictionary for
        :param domains:
        :raises: ValueError if domains are out of bounds or if incompatible with number of parameters
        """
        if len(parameters) != len(domains):
            raise ValueError("Provide as many domains as parameters.")
        for i, domain in enumerate(domains):
            parameter_name = list(parameters.keys())[i]
            scheme_parameter = self.scheme[parameter_name]
            if scheme_parameter.lower_bound >= domain[0] or scheme_parameter.upper_bound <= domain[1]:
                raise ValueError(f"Domains for {parameter_name} are out of parameter bounds")


def plot_gaussian_fit(parameter_distribution: pd.DataFrame, gaussian_fit: pd.DataFrame,
                      color: str = None, fig_label: str = "parameter distributions") -> plt.Figure:
    """
    Plots the gaussian fitting to a parameter distribution. The fit results are returned (i.e. the mean and standard
    deviation that fits best

    :param parameter_distribution: The pandas dataframe containing the parameter
    distribution
    :param ground_truth: the ground truth tissuemodel used in the simulation
    :param color: the color
    used for plotting
    :param label: The label used for the histograms (of none provided recalling the function will plot into the same figure)
    :return: The figure object
    """
    fig = plt.figure(fig_label)
    n_rows = math.ceil(parameter_distribution.shape[1] / 3)

    for i, parameter in enumerate(parameter_distribution.keys()):
        ax = plt.subplot(n_rows, 3, i + 1)

        # Making a histogram
        ax.hist(parameter_distribution[parameter], bins='scott', alpha=0.5, color=color)

        # Plotting the fitted normal distribution as well
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        fitted_mean = gaussian_fit['mean'][parameter]
        fitted_std = gaussian_fit['std'][parameter]
        ax.plot(x, norm.pdf(x, fitted_mean, fitted_std), color=color)

        ax.set_xlabel(r"$\Delta$")
        # plotting ground truth as vertical lines
        ax.vlines(0, 0, 1, transform=ax.get_xaxis_transform(), colors="black")
        ax.set_title(parameter)

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

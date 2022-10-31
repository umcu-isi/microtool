import math
from typing import Dict, List, Tuple

import pandas as pd
from scipy.stats import norm

from microtool.optimize import LossFunction, crlb_loss
from microtool.acquisition_scheme import AcquisitionScheme
from microtool.tissue_model import TissueModel
from copy import copy
import numpy as np
import matplotlib.pyplot as plt


class LossInspector:
    def __init__(self, scheme: AcquisitionScheme, model: TissueModel, noise_var: float,
                 loss_function: LossFunction = crlb_loss, N_samples: int = 100):
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
        self.N_samples = N_samples

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

    def plot(self, parameters: List[Dict[str, int]], domains: List[Tuple[float, float]] = None) -> None:
        """
        Allows for plotting a loss function as a function of 2 or 1 parameter(s) close to a minimum found by
        optimization.

        :param parameters: A dictionary containing a free parameter name as key and the pulse for which you wish to
                           inspect the loss.
        :param domains: The domains for which you wish to inspect the parameters
        :raises: ValueError if domains are out of bounds or if arguments are incompatible
        """

        for parameter in parameters:
            for key, pulse_id in parameter.items():
                # check if the provided parameters are actually in the scheme
                if key not in self.scheme.free_parameters:
                    raise ValueError("The provided acquisition parameter key(s) do not match the provided schemes free "
                                     f"parameters choices are {self.scheme.free_parameter_keys}")

                    # check if provided pulse id is the correct value
                    if pulse_id >= self.scheme.pulse_count or pulse_id < 0:
                        raise ValueError(f"Invalid pulse id provided for {key}.")

        x_optimal = self.scheme.free_parameter_vector / self.scheme.free_parameter_scales

        # extracting the indices of the parameters we want to plot
        parameter_idx = []
        for parameter in parameters:
            parameter_idx.append(self.scheme.get_free_parameter_idx(*parameter.keys(), *parameter.values()))

        parameters_optimal = [x_optimal[i] for i in parameter_idx]
        # make domains if not provided
        if domains:
            self._check_domains(parameters, domains)
            domains = np.array(domains)
        else:
            domains = self._make_domains(parameters)
            domains = np.array(domains)

        # discretizing domains
        domains = np.linspace(domains[:, 0], domains[:, 1], endpoint=True, num=self.N_samples)

        # getting plotting parameters scales to get correct axes later
        scales = []
        for parameter in parameters:
            key = list(parameter.keys())[0]
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
            labels = [list(parameter.keys()) for parameter in parameters]
            ax.set_xlabel(labels[0][0] + "[" + str(*parameters[0].values()) + "]")
            ax.set_ylabel(labels[1][0] + "[" + str(*parameters[1].values()) + "]")
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
            plt.xlabel(list(parameters[0].keys())[0] + " [" + str(parameters[0].values()) + "]")
            plt.ylabel("Loss function")
            plt.legend()
            plt.tight_layout()

    def _make_domains(self, parameters: List[Dict[str, int]]) -> List[Tuple[float, float]]:
        """
        :param parameters: Dictionary with parameter key and pulse id
        :return: Domains of 0.1 around optimum (or at parameter boundary otherwise)
        """
        domains = []
        for parameter in parameters:
            # for default domains we just
            key = list(parameter.keys())[0]
            tissue_parameter = self.scheme[key]
            pulse_id = list(parameter.values())[0]

            lb = tissue_parameter.values[pulse_id] / tissue_parameter.scale - 0.1
            ub = tissue_parameter.values[pulse_id] / tissue_parameter.scale + 0.1
            if tissue_parameter.upper_bound and ub > tissue_parameter.upper_bound:
                ub = tissue_parameter.upper_bound
            if tissue_parameter.lower_bound and lb < tissue_parameter.lower_bound:
                lb = tissue_parameter.lower_bound

            domains.append((lb, ub))
        return domains

    def _check_domains(self, parameters: List[Dict[str, int]], domains: List[Tuple[float, float]]):
        """
        :param parameters: Dictionary for
        :param domains:
        :raises: ValueError if domains are out of bounds or if incompatible with number of parameters
        """
        if len(parameters) != len(domains):
            raise ValueError("Provide as many domains as parameters.")

        for i, domain in enumerate(domains):
            parameter_name = list(parameters[i].keys())[0]
            scheme_parameter = self.scheme[parameter_name]

            lb = scheme_parameter.lower_bound
            ub = scheme_parameter.upper_bound
            if lb is None:
                lb = -np.inf
            if ub is None:
                ub = np.inf

            if lb >= domain[0] or ub <= domain[1]:
                raise ValueError(f"Domains for {parameter_name} are out of parameter bounds")


def plot_parameter_distributions(parameter_distribution: pd.DataFrame, gaussian_fit: pd.DataFrame = None,
                                 color: str = None, fig_label: str = "parameter distributions") -> plt.Figure:
    """
    Plots the parameter distributions. Also a gaussian fitting if its provided.
    Calling this function with the same fig_label argument plot the results in the same figure.

    :param parameter_distribution: The pandas dataframe containing the parameter
    distribution
    :param guassian_fit: The result of the gaussian fitting
    :param color: the color
    used for plotting
    :param label: The label used for the histograms (of none provided recalling the function will plot into the same figure)
    :return: The figure object
    """
    fig = plt.figure(fig_label)

    # if there are less than 3 plots make that the number of columns
    n_tissue_parameters = min(parameter_distribution.shape)

    if n_tissue_parameters < 3:
        ncols = n_tissue_parameters
    else:
        ncols = 3

    n_rows = math.ceil(parameter_distribution.shape[1] / ncols)

    for i, parameter in enumerate(parameter_distribution.keys()):
        ax = plt.subplot(n_rows, ncols, i + 1)

        # Making a histogram
        ax.hist(parameter_distribution[parameter], bins='scott', alpha=0.5, color=color)

        # Plotting the fitted normal distribution as well if provided
        if gaussian_fit is not None:
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


def plot_acquisition_parameters(scheme: AcquisitionScheme, title: str = None) -> plt.Figure:
    """
    Makes subplots of all the acquisition parameters
    :param scheme:
    :return: matplotlib figure
    """
    n_par = len(scheme)
    if n_par < 3:
        n_cols = n_par
    else:
        n_cols = 3

    n_rows = math.ceil(n_par / n_cols)
    fig = plt.figure(title)
    for i, parameter in enumerate(scheme):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        y = scheme[parameter].values
        x = np.arange(len(y)) + 1
        ax.plot(x, y, '.')
        plt.xticks(range(scheme.pulse_count), np.array(range(scheme.pulse_count)) + 1)
        ax.set_xlabel("Measurement")
        ax.set_ylabel(parameter + " [{}]".format(scheme[parameter].unit))
    plt.suptitle(title)
    plt.tight_layout()
    return fig


def plot_signal(scheme: AcquisitionScheme, model: TissueModel, fig_title: str = None, fig: plt.Figure = None):
    # Signal plotting
    if fig is None:
        if fig_title is None:
            fig = plt.figure("signal plot")
        else:
            fig = plt.figure(fig_title)

    plt.title('Signal plot')
    plt.plot(model(scheme), '.')
    plt.xlabel('Measurement')
    plt.xticks(range(scheme.pulse_count), np.array(range(scheme.pulse_count)) + 1)
    plt.ylabel(r'$S/S_0$')
    plt.legend()
    plt.tight_layout()
    return fig

from .optimize import LossFunction
from .acquisition_scheme import AcquisitionScheme
from .tissue_model import TissueModel
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt


class LossInspector:
    def __init__(self, loss_function: LossFunction, scheme: AcquisitionScheme, model: TissueModel, noise_var: float):
        """

        :param loss_function: The loss function we wish to inspect
        :param scheme: The (optimized) acquisition scheme for which we wish to inspect the loss
        :param model: The tissuemodel under investigation
        :param noise_var: The noise variance estimation
        """
        self.scheme = deepcopy(scheme)
        self.scheme_original = deepcopy(scheme)
        self.model = model
        self.noise_var = noise_var
        self.loss_function = loss_function

    def compute_loss(self, x: np.ndarray) -> float:
        """

        :param x: The scaled parameters for which we wish to know the loss
        :return:
        """
        tissue_scales = self.model.get_scales()
        tissue_include = self.model.get_include()
        acq_scales = self.scheme.get_free_parameter_scales()
        # we update the scheme to compute the loss altough this changes not the scheme the user supplied
        self.scheme.set_free_parameters(x * acq_scales)
        jac = self.model.jacobian(self.scheme)
        return self.loss_function(jac, tissue_scales, tissue_include, self.noise_var)

    def plot(self, domain_lengths: np.ndarray,
             plot_parameters: np.ndarray) -> None:
        """
        Allows for plotting a loss function as a function of 2 or 1 parameter(s) close to a minimum found by optimization.

        :param result: The optimization result we wish to plot.
        :param fun: the loss function used during optimization
        :param domain_lengths: The length of the parameter domains we wish to visualize
        :param plot_parameters: The index of the parameters we wish to visualize
        :return: None, instantiates a matplotlib.pyplot figure.
        """
        # checking if things are okay
        if len(plot_parameters) > 2:
            raise ValueError("Cant plot loss a function of more than 2 parameters")

        # extracting the minimum

        x_optimal = self.scheme_original.get_free_parameters() / self.scheme_original.get_free_parameter_scales()
        # extracting the parameters of interest value at the minimum
        parameters_optimal = x_optimal[plot_parameters]
        # discretizing the domain using the domain length
        # TODO: Ensure that domain is in bounds!
        domains = np.linspace(parameters_optimal - 0.5 * domain_lengths, parameters_optimal + 0.5 * domain_lengths,
                              endpoint=True)
        # TODO: Plot optimal parameter value using vline or plane

        # 3d plots for 2 investigated parameters
        if len(plot_parameters) == 2:
            def loss(x1, x2):
                # return loss given two parameters we are changing and other parameters left constant
                params = x_optimal
                params[plot_parameters[0]] = x1
                params[plot_parameters[1]] = x2
                return self.compute_loss(params)

            # Vectorization so we can pass the discretized domains and compute on all values
            vloss = np.vectorize(loss)
            # make a meshgrid out of the two changing parameters so we can compute loss on the entire grid
            X1, X2 = np.meshgrid(domains[:, 0], domains[:, 1])

            # make a plot of this loss versus the parameters
            plt.figure()
            ax = plt.axes(projection='3d')
            Z = vloss(X1, X2)
            ax.plot_surface(X1, X2, Z)
        else:
            # Normal plot if investigating 1 parameter
            def loss(x1):
                # return loss given x1
                params = x_optimal
                params[plot_parameters[0]] = x1
                return self.compute_loss(params)

            # vectorization if so we can pass the domain
            vloss = np.vectorize(loss)
            plt.figure()
            plt.plot(domains[:, 0], vloss(domains[:, 0]))


def wrap_loss(loss_function: LossFunction, scheme: AcquisitionScheme, model: TissueModel, noise_var: float) -> callable:
    """

    :param loss_function: microtool.optimize LossFunction, might change signature in future...
    :param scheme:
    :param model:
    :param noise_var:
    :return:
    """
    scheme = deepcopy(scheme)
    aq_scales = scheme.get_free_parameter_scales()
    tissue_scales = model.get_scales()
    tissue_include = model.get_include()

    def calc_loss(x: np.ndarray):
        scheme.set_free_parameters(x * aq_scales)
        jac = model.jacobian(scheme)
        return loss_function(jac, tissue_scales, tissue_include, noise_var)

    return calc_loss

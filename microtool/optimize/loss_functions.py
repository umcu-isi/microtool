from abc import abstractmethod, ABC
from typing import Callable

import numpy as np
from numba import njit

from ..acquisition_scheme import AcquisitionScheme
from ..tissue_model import TissueModel
from ..utils.math import cartesian_product, diagonal

# We use the machine precision of numpy floats to determine if we can invert a matrix without introducing a large
# numerical error
CONDITION_THRESHOLD = 1 / np.finfo(np.float64).eps

# Arbitrary high cost value for ill conditioned matrices
ILL_COST = 1e30

InformationFunction = Callable[[np.ndarray, np.ndarray, float], np.ndarray]


@njit
def fisher_information_gauss(jac: np.ndarray, _signal: np.ndarray, noise_var: float) -> np.ndarray:
    """
    Calculates the Fisher information matrix, assuming Gaussian noise.
    This is the sum of the matrices of squared gradients, for all samples, divided by the noise variance.
    See equation A2 in Alexander, 2008 (DOI 0.1002/mrm.21646)

    :param jac: An N×M Jacobian matrix, where N is the number of samples and M is the number of parameters.
    :param _signal: The signal, not used.
    :param noise_var: The noise variance
    :return: The M×M information matrix.
    """
    return (1 / noise_var) * jac.T @ jac


@njit
def fisher_information_rice(jac: np.ndarray, signal: np.ndarray, noise_var: float) -> np.ndarray:
    """
    Calculates the fisher information matrix assuming Rician noise.The integral term can be approximated up to 
    4% as reported in https://doi.org/10.1109/TIT.1967.1054037 (eq. 8)

    :param jac: An N×M Jacobian matrix, where N is the number of samples and M is the number of parameters.
    :param signal: The signal
    :param noise_var: The noise variance
    :return: The MxN information matrix
    """

    # Approximating the integral, which does not have a closed form expression.
    sigma = np.sqrt(noise_var)
    z = 2 * signal * (sigma + signal) / (sigma + 2 * signal)

    # Computing the cross term derivatives
    derivative_term = cartesian_product(jac)

    # TODO: there is probably some mistake here, since:
    #  1) this can become negative (e.g. with jac=1, signal=2, noise var=1).
    #  2) z and signal have different units.
    return (1 / noise_var ** 2) * np.sum(derivative_term * (z - signal ** 2), axis=-1)


class LossFunction(ABC):
    """
    Base class for loss functions, just to ensure call signature constant between different loss functions
    """

    @abstractmethod
    def __call__(self, jac: np.ndarray, signal: np.ndarray, noise_var: float) -> float:
        """
        Computes a loss value

        :param jac: the Jacobian of the signal w.r.t. the tissue parameters (should be preprocessed)
        :param signal: the actual signal
        :param noise_var: the variance of the noise distribution
        :return: loss
        """
        raise NotImplementedError()


class CrlbBase(LossFunction, ABC):
    """
    Base class for CRLB-based loss functions. This bass class is made because checking of the fisher information matrix
    is the same for all crlb based loss functions. We ensure that derived classes can actually
    compute crlb by abstractmethod.
    """

    def __init__(self, information_func: InformationFunction):
        """
        Loading the function for computing the Fisher information

        :param information_func: Fisher information function
        """
        self.information_func = information_func

    def __call__(self, jac: np.ndarray, signal: np.ndarray, noise_var: float) -> float:
        """
        Computes the loss using the compute crlb method implemented on the child classes.
        If the fisher information matrix is ill conditioned we return a high loss value.

        :param jac: The preprocessed Jacobian
        :param signal: The signal
        :param noise_var: The noise variance
        :return: Loss value
        """
        # Calculate the Fisher information matrix on the Jacobian.
        # A properly scaled matrix gives an interpretable condition number and a
        # more robust inverse.
        information = self.information_func(jac, signal, noise_var)

        # An ill-conditioned information matrix should result in a high loss.
        if np.linalg.cond(information) > CONDITION_THRESHOLD:
            return ILL_COST

        return self.compute_crlb(information)

    @staticmethod
    @abstractmethod
    def compute_crlb(information: np.ndarray) -> float:
        """
        Computes the cramer roa lower bound based loss given the fisher information (assumed well conditioned)

        :param information: A well conditioned fisher information matrix
        :return: loss
        """
        raise NotImplementedError()


class CrlbInversion(CrlbBase):
    @staticmethod
    @njit
    def compute_crlb(information: np.ndarray) -> float:
        """
        Computes the crlb-based loss by inverting the information matrix

        :param information: Fisher information matrix of shape (N,N) where N is number of parameters in model
        :return: crlb-based loss
        """
        # computing the CRLB for every parameter trough inversion of Fisher matrix and getting the diagonal
        crlb = diagonal(np.linalg.inv(information))
        return float(np.sum(crlb))

    def compute_crlb_individual(self, jac: np.ndarray, signal: np.ndarray, noise_var: float) -> np.ndarray:
        """
        Computes the crlb of the individual parameters
        :param jac: the Jacobian of the signal w.r.t. the tissue parameters (should be preprocessed)
        :param signal: the actual signal
        :param noise_var: The noise variance
        :return: crlb from matrix diagonal
        """
        information = self.information_func(jac, signal, noise_var)
        return diagonal(np.linalg.inv(information))


class CrlbEigenvalues(CrlbBase):
    @staticmethod
    @njit
    def compute_crlb(information: np.ndarray) -> float:
        """
        Computes the crlb-based loss using eigenvalues of the information matrix. For proof see appendix of notes

        :param information: Fisher information matrix of shape N,N
        :return: The crlb based loss
        """
        # Not actually the crlb but just the eigenvalues and sum does correspond to sum of crlb's
        crlb = (1 / np.linalg.eigvalsh(information))
        return crlb.sum()


def compute_crlbs(scheme: AcquisitionScheme, model: TissueModel, noise_var: float) -> np.ndarray:
    """
    Function for computing the cramer rao lower bounds of the
    :param scheme: AcquisitionScheme instance utilized for crlbs computation
    :param model: TissueModel instance utilized for crlbs computation
    :param noise_var: The noise variance
    :return: the crlb based loss computed from the acquisition scheme and tissue model characteristics
    """
    jac = model.scaled_jacobian(scheme)
    signal = model(scheme)
    return inversion_gauss.compute_crlb_individual(jac, signal, noise_var)


def compute_loss(scheme: AcquisitionScheme,
                 model: TissueModel,
                 noise_var: float,
                 loss: LossFunction) -> float:
    """
    Function for computing the loss given the following parameters

    :param model: The tissue model for which you wish to know the loss
    :param scheme: The acquisition scheme for which you wish to know the loss
    :param noise_var: The scaling factor of the noise distribution
    :param loss: The loss function
    :return: The loss value associated with this model and scheme
    """
    jac = model.scaled_jacobian(scheme)
    signal = model(scheme)
    return loss(jac, signal, noise_var)


def scipy_loss(x: np.ndarray, scheme: AcquisitionScheme, model: TissueModel, noise_variance: float,
               loss: LossFunction, scaling_factor: float = 1.0) -> float:
    """
    Wraps the compute loss function, where we use the API of AcquisitionScheme to set the acquisition parameters to
    the optimizers' desired values.

    :param x: The acquisition parameter vector provided by scipy optimization (so scaled values in an np.ndarray)
    :param scheme: The AcquisitionScheme
    :param model: The TissueModel
    :param noise_variance: The noise shape parameter (variance for gaussian)
    :param loss: The loss function to be used in optimization
    :param scaling_factor: The factor by which the output of the loss function is scaled
    :return: The loss value associated with parameters above.
    """
    # updating the scheme with the optimizers search values
    acquisition_parameter_scales = scheme.free_parameter_scales
    scheme.set_free_parameter_vector(x * acquisition_parameter_scales)
    return scaling_factor * compute_loss(scheme, model, noise_variance, loss)


# ---------- Current loss function selection.
inversion_gauss = CrlbInversion(fisher_information_gauss)
inversion_rice = CrlbInversion(fisher_information_rice)
eigenvalue_gauss = CrlbEigenvalues(fisher_information_gauss)
eigenvalue_rice = CrlbEigenvalues(fisher_information_rice)

default_loss = eigenvalue_gauss

from typing import Sequence, Callable

import numpy as np
import scipy
from numba import njit

from microtool.acquisition_scheme import AcquisitionScheme
from microtool.tissue_model import TissueModel
# We use the machine precision of numpy floats to determine if we can invert a matrix without introducing a large
# numerical error
from microtool.utils.fisher_information import cartesian_product

CONDITION_THRESHOLD = 1 / np.finfo(np.float64).eps

# Arbitrary high cost value for ill conditioned matrices
ILL_COST = 1e9

InformationFunction = Callable[[np.ndarray, np.ndarray, float], np.ndarray]


@njit
def fisher_information_gauss(jac: np.ndarray, signal: np.ndarray, noise_var: float) -> np.ndarray:
    """
    Calculates the Fisher information matrix, assuming Gaussian noise.
    This is the sum of the matrices of squared gradients, for all samples, divided by the noise variance.
    See equation A2 in Alexander, 2008 (DOI 0.1002/mrm.21646)

    :param signal:
    :param jac: An N×M Jacobian matrix, where N is the number of samples and M is the number of parameters.
    :param noise_var: Noise variance.
    :return: The M×M information matrix.
    """
    # TODO: Add Rician noise version, as explained in the appendix to Alexander, 2008 (DOI 0.1002/mrm.21646).
    return (1 / noise_var) * jac.T @ jac


@njit
def fisher_information_rice(jac: np.ndarray, signal: np.ndarray, noise_var: float) -> np.ndarray:
    """
    The integral term can be approximated up to 4% as reported in https://doi.org/10.1109/TIT.1967.1054037
    :param jac:
    :param noise_var:
    :return:
    """

    # Approximating the integral without closed form
    sigma = np.sqrt(noise_var)
    Z = 2 * signal * (sigma + signal) / (sigma + 2 * signal)

    # Computing the cross term derivatives
    derivative_term = cartesian_product(jac)

    return (1 / noise_var ** 2) * np.sum(derivative_term * (Z - signal ** 2), axis=-1)


class LossFunction:
    def __call__(self, jac: np.ndarray, signal: np.ndarray, scales: Sequence[float], include: Sequence[bool],
                 noise_var: float) -> float:
        raise NotImplementedError()

    @staticmethod
    def preprocess_jacobian(jac: np.ndarray, scales: Sequence[float], include: Sequence[bool]):
        # Extracting the jacobian w.r.t the included parameters only
        # casting to numpy array if not done already
        include = np.array(include)
        scales = np.array(scales)
        N_measurements = jac.shape[0]
        jacobian_mask = np.tile(include, (N_measurements, 1))
        jac_included = jac[jacobian_mask].reshape((N_measurements, -1))

        # Scaling the jacobian
        jac_rescaled = jac_included * scales[include]
        return jac_rescaled


class CrlbBase(LossFunction):
    def __init__(self, information_func: InformationFunction):
        self.information_func = information_func

    def __call__(self, jac: np.ndarray, signal: np.ndarray, scales: Sequence[float], include: Sequence[bool],
                 noise_var: float) -> float:
        raise NotImplementedError()


class CrlbInversion(CrlbBase):
    def __call__(self, jac: np.ndarray, signal: np.ndarray, scales: Sequence[float], include: Sequence[bool],
                 noise_var: float) -> float:
        """
        A different way to compute the loss where we do matrix inversion and rescale the crlb after computation.

        :param jac:
        :param scales:
        :param include:
        :param noise_var:
        :return:
        """
        # preprocessing
        jac_rescaled = self.preprocess_jacobian(jac, scales, include)

        # Calculate the Fisher information matrix on the rescaled Jacobian.
        # A properly scaled matrix gives an interpretable condition number and a
        # more robust inverse.
        information = self.information_func(jac_rescaled, signal, noise_var)

        # An ill-conditioned information matrix should result in a high loss.
        if np.linalg.cond(information) > CONDITION_THRESHOLD:
            return ILL_COST

        # computing the CRLB for every parameter trough inversion of Fisher matrix and getting the diagonal

        crlb = scipy.linalg.inv(information).diagonal()

        # rescaling the CRLB (we square the scales since the information is computed by matrix product of scaled fisher
        # information and so in essence we scaled twice). Also we multiply since the inversion effectively inverted the
        # scaling as well.
        return float(np.sum(crlb * np.array(scales)[np.array(include)] ** 2))


class CrlbEigenvalues(CrlbBase):
    def __call__(self, jac: np.ndarray, signal: np.ndarray, scales: Sequence[float], include: Sequence[bool],
                 noise_var: float) -> float:
        """
        Objective function for minimizing the total parameter variance (Cramer-Rao lower bounds), as defined in Alexander,
        2008 (DOI https://doi.org/10.1002/mrm.21646)

        :param signal:
        :param jac: An N×M Jacobian matrix, where N is the number of samples and M is the number of parameters.
        :param scales: A sequence with M parameter scales.
        :param include: A boolean sequence specifying which parameters should be included in the loss.
        :param noise_var: Noise variance.
        :return: Estimated total weighted parameter variance.
        """
        # preprocessing
        jac_rescaled = self.preprocess_jacobian(jac, scales, include)

        # Calculate the Fisher information matrix on the rescaled Jacobian.
        # A properly scaled matrix gives an interpretable condition number and a
        # more robust inverse.
        information = self.information_func(jac_rescaled, signal, noise_var)

        # An ill-conditioned information matrix should result in a high loss.
        if np.linalg.cond(information) > CONDITION_THRESHOLD:
            return ILL_COST

        # The loss is the sum of the cramer roa lower bound for every tissue parameter. This is the same as the sum of
        # the reciprocal eigenvalues of the information matrix, which can be calculated at a lower computational cost
        # because of symmetry of the information matrix.
        return (1 / np.linalg.eigvalsh(information)).sum()


def compute_loss(scheme: AcquisitionScheme,
                 model: TissueModel,
                 noise_var: float,
                 loss: LossFunction) -> float:
    """
    Function for computing the loss given the following parameters

    :param model: The tissuemodel for which you wish to know the loss
    :param scheme: The acquisition scheme for which you wish to know the loss
    :param noise_var:
    :param loss:
    :return:
    """
    jac = model.jacobian(scheme)
    signal = model(scheme)
    return loss(jac, signal, model.scales, model.include, noise_var)


def check_initial_scheme(scheme: AcquisitionScheme,
                         model: TissueModel,
                         noise_var: float,
                         loss: LossFunction) -> float:
    """
    Function for computing the loss given the following parameters

    :param model: The tissuemodel for which you wish to know the loss
    :param scheme: The acquisition scheme for which you wish to know the loss
    :param noise_var:
    :param loss:
    :return:
    """
    jac = model.jacobian(scheme)

    include = np.array(model.include)

    # the parameters we included for optimization but to which the signal is insensitive
    # (jac is signal derivative for all parameters)
    insensitive_parameters = include & np.all(jac == 0, axis=0)

    if np.any(insensitive_parameters):
        raise RuntimeError(
            f"Initial AcquisitionScheme error: the parameters {np.array(model.parameter_names)[insensitive_parameters]} have a zero signal derivative for all measurements. "
            f"Optimizing will not result in a scheme that better estimates these parameters. "
            f"Exclude them from optimization if you are okay with that.")

    output = compute_loss(scheme, model, noise_var, loss)
    if output >= ILL_COST:
        raise RuntimeError(
            f"Initial AcquisitionScheme error: the initial acquisition scheme results in ill conditioned fisher information matrix, possibly due to model degeneracy. "
            f"Try a different initial AcquisitionScheme, or alternatively simplify your TissueModel.")
    return output


def scipy_loss(x: np.ndarray, scheme: AcquisitionScheme, model: TissueModel, noise_variance: float,
               loss: LossFunction):
    """

    :param x:
    :param scheme:
    :param model:
    :param noise_variance:
    :param loss:
    :return:
    """
    # updating the scheme with the optimizers search values
    acquisition_parameter_scales = scheme.free_parameter_scales
    scheme.set_free_parameter_vector(x * acquisition_parameter_scales)
    return compute_loss(scheme, model, noise_variance, loss)


# ---------- Current loss function selection.
inversion_gauss = CrlbInversion(fisher_information_gauss)
inversion_rice = CrlbInversion(fisher_information_rice)
eigenvalue_gauss = CrlbEigenvalues(fisher_information_gauss)
eigenvalue_rice = CrlbEigenvalues(fisher_information_rice)

default_loss = eigenvalue_rice

from itertools import product
from typing import Callable

import numpy as np
from numba import njit

from ..acquisition_scheme import AcquisitionScheme
from ..tissue_model import TissueModel
from ..utils.math import cartesian_product, diagonal
from ..utils.unit_registry import cast_to_ndarray
from scipy.special import ive

# As a rule of thumb, if the condition number is 10^k, then you may lose up to k digits of accuracy on top of what would
# be lost to the numerical method. Allow loosing 1% of 32-bit floating point precision (approx. 5 of 7 digits
# precision):
CONDITION_THRESHOLD = 1e-2 / np.finfo(np.float32).eps

# Arbitrary high cost value for ill conditioned matrices
ILL_LOSS = 1e30

InformationFunction = Callable[[np.ndarray, np.ndarray, float], np.ndarray]


def _sum_over_parameter_values(model: TissueModel, evaluate: Callable):
    listed_parameters = [
        (index, parameter)
        for index, parameter in enumerate(model.values())
        if parameter.value_list is not None
    ]
    if not listed_parameters:
        return evaluate()
    if any(not parameter.value_list for _, parameter in listed_parameters):
        raise ValueError("TissueParameter.value_list must contain at least one value.")

    original_parameters = np.array(model.scaled_parameter_vector, copy=True)
    listed_values = [
        [
            float(cast_to_ndarray(value / parameter.scale))
            for value in parameter.value_list
        ]
        for _, parameter in listed_parameters
    ]

    results = []
    try:
        for values in product(*listed_values):
            scenario_parameters = original_parameters.copy()
            for (index, _), value in zip(listed_parameters, values):
                scenario_parameters[index] = value
            model.set_scaled_parameters(scenario_parameters)
            results.append(evaluate())
    finally:
        model.set_scaled_parameters(original_parameters)

    return np.sum(results, axis=0)


@njit
def fisher_information_gauss(jac: np.ndarray, _signal: np.ndarray, noise_var: float) -> np.ndarray:
    """
    Calculates the Fisher information matrix, assuming Gaussian noise.
    This is the sum of the matrices of squared gradients, for all samples, divided by the noise variance.
    See equation A2 in Alexander, 2008 (DOI 0.1002/mrm.21646)

    :param jac: An N×M Jacobian matrix, where N is the number of measurements and M is the number of parameters.
    :param _signal: Not used.
    :param noise_var: The noise variance.
    :return: The M×M information matrix.
    """
    return (1 / noise_var) * jac.T @ jac


@njit
def fisher_information_rice(jac: np.ndarray, signal: np.ndarray, noise_var: float) -> np.ndarray:
    """
    Calculates the Fisher information matrix assuming Rician noise.The integral term can be approximated up to
    4% as reported in https://doi.org/10.1109/TIT.1967.1054037 (eq. 8)

    :param jac: An N×M Jacobian matrix, where N is the number of measurements and M is the number of parameters.
    :param signal: An N-element measurement vector.
    :param noise_var: The noise variance.
    :return: The MxN information matrix
    """

    # Approximating the integral, which does not have a closed form expression.
    sigma = np.sqrt(noise_var)
    z = 2 * signal * (sigma + signal) / (sigma + 2 * signal)

    # Computing the cross term derivatives
    derivative_term = cartesian_product(jac)


    return (1 / noise_var ** 2) * np.sum(derivative_term * (z - signal ** 2), axis=-1)

def _build_xi_table(theta_max=15.0, h=0.01, n_quad=100):
    """xi(theta) on a uniform grid via Gauss-Legendre. Runs once at import."""
    n = int(round(theta_max / h)) + 1
    thetas = np.linspace(0.0, theta_max, n)
    x_gl, w_gl = np.polynomial.legendre.leggauss(n_quad)
    xi = np.zeros(n)                                  # xi(0) = 0
    for i in range(1, n):
        th = thetas[i]
        lo, hi = max(0.0, th - 12.0), th + 12.0
        m = 0.5 * (hi - lo) * x_gl + 0.5 * (hi + lo)
        w = 0.5 * (hi - lo) * w_gl
        z = m * th
        f = m**3 * np.exp(-0.5 * (m - th) ** 2) * ive(1, z) ** 2 / ive(0, z)
        xi[i] = np.sum(w * f) - th**2
    return thetas[1] - thetas[0], np.clip(xi, 0.0, 1.0)


_H, _XI = _build_xi_table()
_THETA_MAX = _H * (len(_XI) - 1)


@njit(cache=True)
def fisher_information_rice_2(jac, signal, noise_var):
    """
    Rician FIM: I = J^T diag(xi(nu/sigma) / sigma^2) J   (sigma known)

    :param jac:       N x M Jacobian
    :param signal:    N-element noise-free signal vector
    :param noise_var: noise variance sigma^2 (same units as signal^2)
    :return:          M x M information matrix
    """
    N, M = jac.shape
    inv_var = 1.0 / noise_var
    inv_sigma = np.sqrt(inv_var)
    inv_h = 1.0 / _H
    last = len(_XI) - 2
    info = np.zeros((M, M))

    for k in range(N):
        theta = signal[k] * inv_sigma
        if not theta > 0.0:                    # xi(0)=0; also skips NaN
            continue
        if theta >= _THETA_MAX:                # asymptotic tail
            t2 = 1.0 / (theta * theta)
            xi = 1.0 - 0.5 * t2 - 0.25 * t2 * t2
        else:
            p = theta * inv_h
            i = min(int(p), last)
            f = p - i
            xi = _XI[i] + f * (_XI[i + 1] - _XI[i])
        w = xi * inv_var
        for a in range(M):
            wa = w * jac[k, a]
            for b in range(a, M):
                info[a, b] += wa * jac[k, b]

    for a in range(M):
        for b in range(a + 1, M):
            info[b, a] = info[a, b]
    return info


class LossFunction:
    """
    Base class for loss functions to ensure call signature constant between different loss functions.
    """

    def __call__(self, jac: np.ndarray, signal: np.ndarray, noise_var: float) -> float:
        """
        Computes a scalar loss value.

        :param jac: An N×M Jacobian matrix, where N is the number of measurements and M is the number of parameters.
        :param signal: An N-element measurement vector.
        :param noise_var: The noise variance.
        :return: total loss
        """
        raise NotImplementedError()

    def crlb(self, jac: np.ndarray, signal: np.ndarray, noise_var: float) -> np.ndarray:
        """
        Computes the Cramer-Rao lower bound per parameter.

        :param jac: N×M Jacobian matrix, where N is the number of samples and M is the number of parameters.
        :param signal: Measurement vector with N samples.
        :param noise_var: The noise variance.
        :return: Cramer-Rao lower bounds per parameter.
        """
        raise NotImplementedError()


class CrlbLoss(LossFunction):
    """
    Base class for CRLB-based loss functions. This bass class is made because checking of the Fisher information matrix
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
        Computes the sum of the Cramer-Rao lower bounds of the parameters using eigenvalue computation.
        If the Fisher information matrix is ill conditioned, ILL_LOSS is returned.

        :param jac: An N×M Jacobian matrix, where N is the number of measurements and M is the number of parameters.
        :param signal: An N-element measurement vector.
        :param noise_var: The noise variance.
        :return: The sum of the lower bounds on the parameter variance or ILL_LOSS
        """
        # Calculate the Fisher information matrix on the Jacobian.
        # A properly scaled matrix gives an interpretable condition number and a
        # more robust inverse.
        information = self.information_func(jac, signal, noise_var)

        # An ill-conditioned information matrix should result in a high loss.
        if np.linalg.cond(information) > CONDITION_THRESHOLD:
            return ILL_LOSS

        # The sum of the inverse of the eigenvalues is equal to the trace of the inverse.
        return (1 / np.linalg.eigvalsh(information)).sum()

    def crlb(self, jac: np.ndarray, signal: np.ndarray, noise_var: float) -> np.ndarray:
        """
        Computes the Cramer-Rao lower bounds of the individual parameters using matrix inversion.

        :param jac: N×M Jacobian matrix, where N is the number of samples and M is the number of parameters.
        :param signal: Measurement vector with N samples.
        :param noise_var: The noise variance.
        :return: Cramer-Rao lower bounds per parameter.
        """
        information = self.information_func(jac, signal, noise_var)
        return diagonal(np.linalg.inv(information))


def compute_loss(scheme: AcquisitionScheme,
                 model: TissueModel,
                 noise_var: float,
                 loss: LossFunction) -> float:
    """
    Computes the loss for the given acquisition scheme and model. If tissue
    parameters define ``value_list``, the loss is summed over every listed
    value combination.

    :param model: The tissue model for which you wish to know the loss
    :param scheme: The acquisition scheme for which you wish to know the loss
    :param noise_var: The scaling factor of the noise distribution
    :param loss: The loss function
    :return: The loss value associated with this model and scheme
    """
    def evaluate():
        jac = cast_to_ndarray(model.scaled_jacobian(scheme))
        signal = cast_to_ndarray(model(scheme))
        return loss(jac, signal, noise_var)

    return _sum_over_parameter_values(model, evaluate)


def compute_crlb(scheme: AcquisitionScheme,
                 model: TissueModel,
                 noise_var: float,
                 loss: LossFunction) -> np.ndarray:
    """
    Computes the per-parameter CRLB for the given acquisition scheme and
    model. If tissue parameters define ``value_list``, CRLBs are summed over
    every listed value combination.

    :param model: The tissue model for which you wish to know the loss
    :param scheme: The acquisition scheme for which you wish to know the loss
    :param noise_var: The scaling factor of the noise distribution
    :param loss: The loss function
    :return: The loss value associated with this model and scheme
    """
    def evaluate():
        jac = cast_to_ndarray(model.scaled_jacobian(scheme))
        signal = cast_to_ndarray(model(scheme))
        return loss.crlb(jac, signal, noise_var)

    return _sum_over_parameter_values(model, evaluate)


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
    scheme.set_scaled_free_parameter_vector(x)
    return scaling_factor * compute_loss(scheme, model, noise_variance, loss)


# ---------- Current loss function selection.
gauss_loss = CrlbLoss(fisher_information_gauss)
rice_loss = CrlbLoss(fisher_information_rice)
rice_loss_2 = CrlbLoss(fisher_information_rice_2)

default_loss = rice_loss_2

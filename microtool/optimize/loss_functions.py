from typing import Callable

import numpy as np
from numba import njit

from ..acquisition_scheme import AcquisitionScheme
from ..tissue_model import TissueModel
from ..utils.math import cartesian_product, diagonal
from ..utils.unit_registry import cast_to_ndarray

# As a rule of thumb, if the condition number is 10^k, then you may lose up to k digits of accuracy on top of what would
# be lost to the numerical method. Allow loosing 1% of 32-bit floating point precision (approx. 5 of 7 digits
# precision):
CONDITION_THRESHOLD = 1e-2 / np.finfo(np.float32).eps

# Arbitrary high cost value for ill conditioned matrices
ILL_LOSS = 1e30

InformationFunction = Callable[[np.ndarray, np.ndarray, float], np.ndarray]


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

    # TODO: there is probably some mistake here, since:
    #  1) this can become negative (e.g. with jac=1, signal=2, noise var=1).
    #  2) z and signal have different units.
    return (1 / noise_var ** 2) * np.sum(derivative_term * (z - signal ** 2), axis=-1)

@njit(cache=True)
def fisher_information_rice_2(jac: np.ndarray, signal: np.ndarray, noise_var: float) -> np.ndarray:
    """
    Fisher information matrix for Rician noise.

    I_ij = sum_k  xi(nu_k/sigma) / sigma^2 * J_ki * J_kj

    xi(theta) = int_0^inf m^3 exp(-(m-theta)^2/2) * ive(1, m*theta)^2 / ive(0, m*theta) dm - theta^2

    :param jac: N x M Jacobian (N measurements, M parameters)
    :param signal: N-element vector of noise-free signal values (nu)
    :param noise_var: noise variance (sigma^2)
    :return: M x M information matrix
    """
    N, M = jac.shape
    sigma = np.sqrt(noise_var)
    n_grid = 600
    info = np.zeros((M, M))

    for k in range(N):
        theta = signal[k] / sigma

        # xi -> 1 (Gaussian limit) at high SNR, xi(0) = 0
        if theta > 20.0:
            xi = 1.0
        elif theta <= 0.0:
            xi = 0.0
        else:
            lo = max(0.0, theta - 10.0)
            hi = theta + 10.0
            h = (hi - lo) / (n_grid - 1)
            acc = 0.0
            for i in range(n_grid):
                m = lo + i * h
                x = m * theta
                if x < 1e-12:
                    f = 0.0
                else:
                    # exponentially scaled Bessel I0, I1 (A&S 9.8.1-9.8.4)
                    if x < 3.75:
                        t = (x / 3.75) ** 2
                        i0 = 1.0 + t * (3.5156229 + t * (3.0899424 + t * (1.2067492
                             + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))))
                        i1 = x * (0.5 + t * (0.87890594 + t * (0.51498869 + t * (0.15084934
                             + t * (0.02658733 + t * (0.00301532 + t * 0.00032411))))))
                        e = np.exp(-x)
                        ive0 = i0 * e
                        ive1 = i1 * e
                    else:
                        t = 3.75 / x
                        sq = np.sqrt(x)
                        ive0 = (0.39894228 + t * (0.01328592 + t * (0.00225319 + t * (-0.00157565
                               + t * (0.00916281 + t * (-0.02057706 + t * (0.02635537
                               + t * (-0.01647633 + t * 0.00392377)))))))) / sq
                        ive1 = (0.39894228 + t * (-0.03988024 + t * (-0.00362018 + t * (0.00163801
                               + t * (-0.01031555 + t * (0.02282967 + t * (-0.02895312
                               + t * (0.01787654 + t * (-0.00420059)))))))))  / sq
                    f = m ** 3 * np.exp(-0.5 * (m - theta) ** 2) * ive1 ** 2 / ive0

                w_trap = 0.5 if (i == 0 or i == n_grid - 1) else 1.0
                acc += w_trap * f
            xi = acc * h - theta ** 2
            xi = min(max(xi, 0.0), 1.0)   # numerical guard: 0 <= xi <= 1

        w = xi / noise_var
        for a in range(M):
            for b in range(a, M):
                info[a, b] += w * jac[k, a] * jac[k, b]

    # symmetrize
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
    Computes the loss for the given acquisition scheme and model.

    :param model: The tissue model for which you wish to know the loss
    :param scheme: The acquisition scheme for which you wish to know the loss
    :param noise_var: The scaling factor of the noise distribution
    :param loss: The loss function
    :return: The loss value associated with this model and scheme
    """
    jac = cast_to_ndarray(model.scaled_jacobian(scheme))  # Make sure the result is a dimensionless numpy array.
    signal = cast_to_ndarray(model(scheme))  # Make sure the result is a dimensionless numpy array.
    return loss(jac, signal, noise_var)


def compute_crlb(scheme: AcquisitionScheme,
                 model: TissueModel,
                 noise_var: float,
                 loss: LossFunction) -> np.ndarray:
    """
    Computes the loss for the given acquisition scheme and model.

    :param model: The tissue model for which you wish to know the loss
    :param scheme: The acquisition scheme for which you wish to know the loss
    :param noise_var: The scaling factor of the noise distribution
    :param loss: The loss function
    :return: The loss value associated with this model and scheme
    """
    jac = cast_to_ndarray(model.scaled_jacobian(scheme))  # Make sure the result is a dimensionless numpy array.
    signal = cast_to_ndarray(model(scheme))  # Make sure the result is a dimensionless numpy array.
    return loss.crlb(jac, signal, noise_var)


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

default_loss = gauss_loss

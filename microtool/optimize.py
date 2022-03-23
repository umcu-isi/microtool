from typing import Sequence, Callable, Optional

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from microtool.acquisition_scheme import AcquisitionScheme
from microtool.tissue_model import TissueModel


LossFunction = Callable[[np.ndarray, Sequence[float], float], float]


def fisher_information(jac: np.ndarray, noise_var: float) -> np.ndarray:
    """
    Calculates the Fisher information matrix, assuming Gaussian noise.
    This is the sum of the matrices of squared gradients, for all samples, divided by the noise variance.
    See equation A2 in Alexander, 2008 (DOI 0.1002/mrm.21646)

    :param jac: An N×M Jacobian matrix, where N is the number of samples and M is the number of parameters.
    :param noise_var: Noise variance.
    :return: The M×M information matrix.
    """
    # TODO: Add Rician noise version, as explained in the appendix to Alexander, 2008 (DOI 0.1002/mrm.21646).
    return (1 / noise_var) * jac.T @ jac


def crlb_loss(jac: np.ndarray, scales: Sequence[float], noise_var: float) -> float:
    """
    Objective function for minimizing the total parameter variance (Cramer-Rao lower bounds), as defined in Alexander,
    2008 (DOI 0.1002/mrm.21646)

    :param jac: An N×M Jacobian matrix, where N is the number of samples and M is the number of parameters.
    :param scales: An array with M parameter scales.
    :param noise_var: Noise variance.
    :return: Estimated total weighted parameter variance.
    """
    # Calculate the Fisher information matrix on the rescaled Jacobian. The Cramer-Rao lower bound on parameter variance
    # is the inverse of the information matrix. A properly scaled matrix gives an interpretable condition number and a
    # more robust inverse.
    information = fisher_information(jac * scales, noise_var)

    # An ill-conditioned information matrix should result in a high loss.
    if np.linalg.cond(information) > 1e9:
        return 1e9

    # The loss is the sum of the diagonal values of Cramer-Rao lower bound matrix, which is the inverse information
    # matrix; see equation 2 in Alexander, 2008 (DOI 0.1002/mrm.21646). This is the same as the sum of the reciprocal
    # eigenvalues of the information matrix, which can be calculated at a lower computational cost because of symmetry.
    # Rescaling has already been done by multiplying the Jacobian by the parameter scales.
    return (1 / np.linalg.eigvalsh(information)).sum()


def optimize_acquisition_scheme(
        acquisition_scheme: AcquisitionScheme,
        tissue_model: TissueModel,
        noise_var: float,
        loss: LossFunction = crlb_loss,
        method: Optional[str] = None) -> OptimizeResult:
    """
    Optimizes the free parameters in the given MR acquisition scheme such that the loss is minimized.
    The loss function should be of type LossFunction, which takes an N×M Jacobian matrix, an array with M parameter
    scales, and the noise variance, and returns a loss. N is the number of measurements in the acquisition and M is the
    number of tissue parameters.

    :param acquisition_scheme: The MR acquisition scheme to be optimized.
    :param tissue_model: The tissue model of interest.s
    :param noise_var: Noise variance on the MR signal attenuation.
    :param loss: a function of type LossFunction.
    :param method: Type of solver. See the documentation for scipy.optimize.minimize
    :return: A scipy.optimize.OptimizeResult object.
    """
    scales = acquisition_scheme.model_scales(tissue_model)
    x0 = acquisition_scheme.get_free_parameters()

    # Calculating the loss involves passing the new parameters to the acquisition scheme, calculating the tissue model's
    # Jacobian matrix and evaluating the loss function.
    def calc_loss(x: np.ndarray):
        acquisition_scheme.set_free_parameters(x)
        jac = acquisition_scheme.model_jacobian(tissue_model)
        return loss(jac, scales, noise_var)

    result = minimize(calc_loss, x0, method=method)
    if 'x' in result:
        acquisition_scheme.set_free_parameters(result['x'])

    return result

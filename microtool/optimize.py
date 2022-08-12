from copy import deepcopy
from typing import Sequence, Callable, Optional, Union, List, Tuple, Any

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from microtool.acquisition_scheme import AcquisitionScheme
from microtool.optimization_methods import Optimizer
from microtool.tissue_model import TissueModel

# A LossFunction takes an N×M Jacobian matrix, a sequence of M parameter scales, a boolean sequence that specifies which
# parameters should be included in the loss, and the noise variance. It should return a scalar loss.
LossFunction = Callable[[np.ndarray, Sequence[float], Sequence[bool], float], float]


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


def crlb_loss(jac: np.ndarray, scales: Sequence[float], include: Sequence[bool], noise_var: float) -> float:
    """
    Objective function for minimizing the total parameter variance (Cramer-Rao lower bounds), as defined in Alexander,
    2008 (DOI https://doi.org/10.1002/mrm.21646)

    :param jac: An N×M Jacobian matrix, where N is the number of samples and M is the number of parameters.
    :param scales: A sequence with M parameter scales.
    :param include: A boolean sequence specifying which parameters should be included in the loss.
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
    return np.linalg.eigvalsh(information)[include].sum()


def compute_loss(scheme: AcquisitionScheme,
                 model: TissueModel,
                 noise_var: float,
                 loss: LossFunction = crlb_loss) -> float:
    """
    Function for computing the loss given the following parameters

    :param model: The tissuemodel for which you wish to know the loss
    :param scheme: The acquisition scheme for which you whish to know the loss
    :param noise_var:
    :param loss:
    :return:
    """
    model_scales = [value.scale for value in model.values()]
    model_include = [value.optimize for value in model.values()]
    jac = model.jacobian(scheme)
    return loss(jac, model_scales, model_include, noise_var)


def optimize_scheme(scheme: Union[AcquisitionScheme, List[AcquisitionScheme]], model: TissueModel,
                    noise_variance: float,
                    loss: LossFunction = crlb_loss,
                    method: Optional[Union[str, Optimizer]] = None,
                    repeat: int = 1,
                    **options) -> Tuple[AcquisitionScheme, Optional[OptimizeResult]]:
    """
    Optimizes the free parameters in the given MR acquisition scheme such that the loss is minimized.
    The loss function should be of type LossFunction, which takes an N×M Jacobian matrix, an array with M parameter
    scales, and the noise variance. The loss function should return a scalar loss. N is the number of measurements
    in the acquisition and M is the number of tissue parameters.


    :param scheme: The MR acquisition scheme to be optimized. (NOTE: a reference of the scheme is passed,
                    so it will be changed.
    :param model: The tissuemodel for which we want the optimal acquisition scheme.
    :param noise_variance: Noise variance on the MR signal attenuation.
    :param loss: a function of type LossFunction.
    :param method: Type of solver. See the documentation for scipy.optimize.minimize
    :param repeat: Number of times the optimization process is repeated.
    :return: A scipy.optimize.OptimizeResult object.
    """
    M = int(np.sum(np.array(model.include)))

    # Allowing for multiple schemes to be passed as initial schemes
    if not isinstance(scheme, list):
        schemes = [scheme]
    else:
        schemes = scheme

    # Testing if there is at least 1 initial scheme that will have a chance of optimizing
    initial_losses = np.array([compute_loss(scheme, model, noise_variance) for scheme in schemes])
    if (initial_losses >= 1e9).all():
        raise ValueError("The provided initial scheme(s) have ill conditioned loss value(s). Optimization will not "
                         "succeed, please retry with different initial schemes.")

    # Copying the schemes for number repeated optimizations required.
    schemes = [deepcopy(scheme) for scheme in schemes for _ in range(repeat)]

    # Set best_scheme to scheme with the lowest loss value
    best_scheme = schemes[np.argmin(initial_losses)]
    best_loss = np.min(initial_losses)
    best_result = None
    optimized_losses = []
    for scheme in schemes:
        N = len(scheme.free_parameter_vector)
        if M > N:
            raise ValueError(f"The TissueModel has too many degrees of freedom ({M}) to optimize the "
                             f"AcquisitionScheme parameters ({N}) with meaningful result.")

        scales = model.scales
        include = model.include
        acquisition_parameter_scales = scheme.free_parameter_scales
        x0 = scheme.free_parameter_vector / acquisition_parameter_scales
        scaled_bounds = scheme.free_parameter_bounds_scaled
        constraints = scheme.get_constraints()

        # Calculating the loss involves passing the new parameters to the acquisition scheme, calculating the tissue
        # model's Jacobian matrix and evaluating the loss function.
        def calc_loss_scipy(x: np.ndarray):
            scheme.set_free_parameter_vector(x * acquisition_parameter_scales)
            jac = model.jacobian(scheme)
            return loss(jac, scales, include, noise_variance)

        result = minimize(calc_loss_scipy, x0, method=method, bounds=scaled_bounds, constraints=constraints, options=options)
        if 'x' in result:
            scheme.set_free_parameter_vector(result['x'] * acquisition_parameter_scales)

        # check if the optimized scheme is better than the current best scheme and update
        # also save best optimization result.
        current_loss = compute_loss(scheme, model, noise_variance)
        if current_loss < best_loss:
            best_scheme = scheme
            best_loss = current_loss
            best_result = result

        optimized_losses.append(current_loss)

    #TODO: If one of the schemes does optimize but is not the lowest loss of them all, what do we do?
    if (np.array(optimized_losses).reshape(-1, len(initial_losses)) > initial_losses).all():
        raise RuntimeError("Optimization was unsuccesfull, the optimized schemes have higher loss than the initial "
                           "schemes, probably due to choice of optimizer. Please retry with different optimization "
                           "method.")

    return best_scheme, best_result

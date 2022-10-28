from copy import deepcopy
from typing import Sequence, Callable, Optional, Union, List, Tuple, Any, TypeVar, Type

import numpy as np
import scipy
from scipy.optimize import OptimizeResult, minimize

from microtool.acquisition_scheme import AcquisitionScheme
from microtool.optimization_methods import Optimizer
from microtool.tissue_model import TissueModel
from copy import deepcopy

# A LossFunction takes an N×M Jacobian matrix, a sequence of M parameter scales, a boolean sequence that specifies which
# parameters should be included in the loss, and the noise variance. It should return a scalar loss.
LossFunction = Callable[[np.ndarray, Sequence[float], Sequence[bool], float], float]

CONDITION_THRESHOLD = 1e9


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

    # Extracting the jacobian w.r.t the included parameters only
    # casting to numpy array if not done already
    include = np.array(include)
    scales = np.array(scales)
    N_measurements = jac.shape[0]
    jacobian_mask = np.tile(include, (N_measurements, 1))
    jac_included = jac[jacobian_mask].reshape((N_measurements, -1))

    # Scaling the jacobian
    jac_rescaled = jac_included * scales[include]

    # Removing variables that are not influencing the signal
    # jac_rescaled_reset = jac_rescaled[jac_rescaled!=0].reshape(N_measurements,-1)

    # Calculate the Fisher information matrix on the rescaled Jacobian.
    # A properly scaled matrix gives an interpretable condition number and a
    # more robust inverse.

    information = fisher_information(jac_rescaled, noise_var)

    # An ill-conditioned information matrix should result in a high loss.
    if np.linalg.cond(information) > CONDITION_THRESHOLD:
        return CONDITION_THRESHOLD

    # The loss is the sum of the cramer roa lower bound for every tissue parameter. This is the same as the sum of
    # the reciprocal eigenvalues of the information matrix, which can be calculated at a lower computational cost
    # because of symmetry of the information matrix.
    return (1 / np.linalg.eigvalsh(information)).sum()


def crlb_loss_inversion(jac: np.ndarray, scales: Sequence[float], include: Sequence[bool], noise_var: float) -> float:
    """
    A different way to compute the loss where we do matrix inversion and rescale the crlb after computation.

    :param jac:
    :param scales:
    :param include:
    :param noise_var:
    :return:
    """

    # Extracting the jacobian w.r.t the included parameters only
    # casting to numpy array if not done already
    include = np.array(include)
    scales = np.array(scales)
    N_measurements = jac.shape[0]
    jacobian_mask = np.tile(include, (N_measurements, 1))
    jac_included = jac[jacobian_mask].reshape((N_measurements, -1))

    # Scaling the jacobian
    jac_rescaled = jac_included * scales[include]

    # Calculate the Fisher information matrix on the rescaled Jacobian.
    # A properly scaled matrix gives an interpretable condition number and a
    # more robust inverse.
    information = fisher_information(jac_rescaled, noise_var)

    # An ill-conditioned information matrix should result in a high loss.
    if np.linalg.cond(information) > CONDITION_THRESHOLD:
        return CONDITION_THRESHOLD

    # computing the CRLB for every parameter trough inversion of Fisher matrix and getting the diagonal
    crlb = scipy.linalg.inv(information).diagonal()

    # rescaling the CRLB (we square the scales since the information is computed by matrix product of scaled fisher
    # information and so in essence we scaled twice). Also we multiply since the inversion effectively inverted the
    # scaling as well.
    return float(np.sum(crlb * (scales[include] ** 2)))


def compute_loss(scheme: AcquisitionScheme,
                 model: TissueModel,
                 noise_var: float,
                 loss: LossFunction = crlb_loss) -> float:
    """
    Function for computing the loss given the following parameters

    :param model: The tissuemodel for which you wish to know the loss
    :param scheme: The acquisition scheme for which you wish to know the loss
    :param noise_var:
    :param loss:
    :return:
    """
    jac = model.jacobian(scheme)
    return loss(jac, model.scales, model.include, noise_var)


def compute_loss_init(scheme: AcquisitionScheme,
                      model: TissueModel,
                      noise_var: float,
                      loss: LossFunction = crlb_loss) -> float:
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

    output = loss(jac, model.scales, include, noise_var)
    if output > CONDITION_THRESHOLD:
        raise RuntimeError(
            f"Initial AcquisitionScheme error: the initial acquisition scheme results in ill conditioned fisher information matrix, possibly due to model degeneracy. "
            f"Try a different initial AcquisitionScheme, or alternatively simplify you TissueModel.")
    return output


def calc_loss_scipy(x: np.ndarray, scheme: AcquisitionScheme, model: TissueModel, noise_variance: float,
                    loss: LossFunction):
    scales = model.scales
    include = model.include
    acquisition_parameter_scales = scheme.free_parameter_scales
    scheme.set_free_parameter_vector(x * acquisition_parameter_scales)
    jac = model.jacobian(scheme)
    return loss(jac, scales, include, noise_variance)


# A way of type hinting all the derived classes of AcquisitionScheme
AcquisitionType = TypeVar('AcquisitionType', bound=AcquisitionScheme)


def optimize_scheme(scheme: Union[AcquisitionType, List[AcquisitionType]], model: TissueModel,
                    noise_variance: float,
                    loss: LossFunction = crlb_loss,
                    method: Optional[Union[str, Optimizer]] = None,
                    repeat: int = 1,
                    **kwargs) -> Tuple[AcquisitionType, Optional[OptimizeResult]]:
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

    # Checking if optimization options are provided.
    if 'options' in kwargs.keys():
        optimizer_options = kwargs['options']
    else:
        optimizer_options = None

    # Allowing for multiple schemes to be passed as initial schemes
    if not isinstance(scheme, list):
        schemes = [scheme]
    else:
        schemes = scheme

    # Testing if there is at least 1 initial scheme that will have a chance of optimizing
    initial_losses = np.array([compute_loss_init(scheme, model, noise_variance, loss=loss) for scheme in schemes])

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
        scipy_loss_args = (scheme, model, noise_variance, loss)
        if method == 'differential_evolution':
            # converting dictionary constraints of microtool tissuemodel to scipy NonLinear constraint
            # constraints = scipy.optimize.NonlinearConstraint(constraints['fun'],0,np.inf)
            result = differential_evolution(calc_loss_scipy, bounds=scaled_bounds,
                                            args=scipy_loss_args,
                                            x0=x0, workers=-1, disp=True, updating='deferred')
        elif method == 'dual_annealing':
            result = dual_annealing(calc_loss_scipy, scaled_bounds, scipy_loss_args, x0=x0)
        else:
            result = minimize(calc_loss_scipy, x0, args=scipy_loss_args,
                              method=method, bounds=scaled_bounds, constraints=constraints,
                              options=optimizer_options)

        if 'x' in result:
            scheme.set_free_parameter_vector(result['x'] * acquisition_parameter_scales)

        # check if the optimized scheme is better than the current best scheme and update
        # also save best optimization result.
        current_loss = compute_loss(scheme, model, noise_variance, loss=loss)
        if current_loss < best_loss:
            best_scheme = scheme
            best_loss = current_loss
            best_result = result

        if not best_result["success"]:
            print(result)
            raise RuntimeError(
                "Optimization procedure was unsuccesfull. "
                "Possible solutions include but are not limited to: Changing the "
                "optimizer setings, changing the optimization method or changing the initial scheme to a more suitable one."
                "If you are using a scipy optimizer its settings can be changed by passing options to this function. "
                "If you are using a microtool optimization method please consult the optimization_methods module for more details.")

        optimized_losses.append(current_loss)

    # TODO: If one of the schemes does optimize but is not the lowest loss of them all, what do we do?
    if (np.array(optimized_losses).reshape(-1, len(initial_losses)) > initial_losses).all():
        raise RuntimeError("Optimization was unsuccesfull, the optimized schemes have higher loss than the initial "
                           "schemes, probably due to choice of optimizer method and or settings. "
                           "Please retry with different optimization method.")

    return best_scheme, best_result

import warnings
from copy import deepcopy
from typing import Optional, Union, Tuple, TypeVar, List, Dict

import numpy as np
from scipy.optimize import OptimizeResult, minimize, differential_evolution, Bounds

from .loss_functions import compute_loss, scipy_loss, LossFunction, default_loss, ILL_COST
from .methods import Optimizer
from ..acquisition_scheme import AcquisitionScheme
from ..constants import ConstraintTypes
from ..tissue_model import TissueModel

# A way of type hinting all the derived classes of AcquisitionScheme
AcquisitionType = TypeVar('AcquisitionType', bound=AcquisitionScheme)


def optimize_scheme(scheme: AcquisitionType, model: TissueModel,
                    noise_variance: float,
                    loss: LossFunction = default_loss,
                    loss_scaling_factor: float = 1.0,
                    method: Optional[Union[str, Optimizer]] = "differential_evolution",
                    solver_options: dict = None) -> Tuple[AcquisitionType, Optional[OptimizeResult]]:
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
    :param loss_scaling_factor: Can be used to scale the loss function to order 1.0 if you notice extreme values.
    :param method: Type of solver. See the documentation for scipy.optimize.minimize
    :param solver_options: Options specific to the solver, provided as a dictionary with the options as keywords, see
                            the documentation of the solver for the keywords.
    :return: A scipy.optimize.OptimizeResult object.
    """
    # setting to empty dict because unpack operation is required later
    if solver_options is None:
        solver_options = {}

    # Checking the initial scheme and model
    check_degrees_of_freedom(scheme, model)
    check_insensitive(scheme, model)
    initial_loss = compute_loss(scheme, model, noise_variance, loss)
    check_ill_conditioned(initial_loss)

    # Copying the scheme because acquisition parameters are updated during optimization
    scheme_copy = deepcopy(scheme)

    # getting all the parameters needed for scipy optimization
    x0 = scheme_copy.x0

    scaled_bounds = scheme_copy.free_parameter_bounds_scaled
    scipy_bounds = bounds_tuple2scipy(scaled_bounds)
    constraints = scheme_copy.constraint_list

    # The parameters required to fully define the loss function.
    scipy_loss_args = (scheme_copy, model, noise_variance, loss, loss_scaling_factor)

    if method == 'differential_evolution':
        # constraints are formatted different for this optimization method
        if constraints is None:
            constraints = ()

        result = differential_evolution(scipy_loss, bounds=scipy_bounds,
                                        args=scipy_loss_args,
                                        x0=x0, workers=-1, disp=True, updating='deferred', constraints=constraints,
                                        polish=False, **solver_options)
    else:
        result = minimize(scipy_loss, x0, args=scipy_loss_args,
                          method=method, bounds=scipy_bounds, constraints=constraints,
                          options=solver_options)

    # update the scheme_copy to the result found by the optimizer
    if 'x' in result:
        x = result['x']
        scheme_copy.set_free_parameter_vector(result['x'] * scheme_copy.free_parameter_scales)
    else:
        raise RuntimeError("No suitable solution was found?")

    # check if the optimized scheme is better than the initial scheme
    current_loss = compute_loss(scheme_copy, model, noise_variance, loss)
    if current_loss > initial_loss:
        raise RuntimeError("Loss increased during optimization, try a different optimization method.")

    check_constraints_satisfied(x, constraints)

    warn_early_termination(result)

    return scheme_copy, result


def bounds_tuple2scipy(microtool_bounds: List[Tuple[float, float]]) -> Bounds:
    lb_array = np.array([bound[0] for bound in microtool_bounds])
    ub_array = np.array([bound[1] for bound in microtool_bounds])
    # keep feasible is required to ensure that the optimizer doesn't step out of the search space
    return Bounds(lb_array, ub_array, keep_feasible=True)


def warn_early_termination(result: OptimizeResult):
    if not result["success"]:
        print(result)
        warnings.warn(
            "Optimization procedure was unsuccessful. "
            "Possible solutions include but are not limited to: Changing the "
            "optimizer setings, changing the optimization method or changing the initial scheme to a more suitable one."
            "If you are using a scipy optimizer its settings can be changed by passing options to this function. "
            "If you are using a microtool optimization method please consult the optimization_methods module for more details.")


def check_ill_conditioned(loss_value: float):
    if loss_value == ILL_COST:
        raise RuntimeError(
            f"Initial AcquisitionScheme error: the initial acquisition scheme results in ill conditioned fisher "
            f"information matrix, possibly due to model degeneracy. "
            f"Try a different initial AcquisitionScheme, or alternatively simplify your TissueModel.")


def check_insensitive(scheme: AcquisitionScheme, model: TissueModel):
    jac = model.jacobian(scheme)
    # the parameters we included for optimization but to which the signal is insensitive
    # (jac is signal derivative for all parameters)
    insensitive_parameters = np.all(jac == 0, axis=0)
    if np.any(insensitive_parameters):
        raise ValueError(
            f"Initial AcquisitionScheme error: the parameters {np.array(model.parameter_names)[model.include_optimize][insensitive_parameters]} have a zero signal derivative for all measurements. "
            f"Optimizing will not result in a scheme that better estimates these parameters. "
            f"Exclude them from optimization if you are okay with that.")


def check_degrees_of_freedom(scheme: AcquisitionScheme, model: TissueModel):
    M = int(np.sum(np.array(model.include_optimize)))
    N = len(scheme.free_parameter_vector)
    if M > N:
        raise ValueError(f"The TissueModel has too many degrees of freedom ({M}) to optimize the "
                         f"AcquisitionScheme parameters ({N}) with meaningful result.")


def check_constraints_satisfied(x: np.ndarray, constraints: Dict[str, ConstraintTypes]):
    """
    :param x: The scipy parameter vector
    :param constraints: The constraints dictionary
    :raises: RunTimeError if any of the constraints are not satisfied for parameter vector x
    """
    error_msg = ""
    for key, constraint in constraints.items():
        value = constraint.fun(x)

        lb_c = value < constraint.lb
        ub_c = value > constraint.ub

        if np.any(lb_c) or np.any(ub_c):
            error_msg += f"The {key} constraint is not satisfied."

    if error_msg != "":
        raise RuntimeError(error_msg)

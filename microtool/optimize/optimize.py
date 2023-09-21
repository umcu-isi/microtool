import logging
import os
import warnings
from copy import deepcopy
from datetime import datetime
from typing import Optional, Union, Tuple, TypeVar, List, Dict

import numpy as np
from scipy.optimize import OptimizeResult, minimize, differential_evolution, Bounds

from .loss_functions import compute_loss, scipy_loss, LossFunction, default_loss, ILL_COST, compute_crlbs
from .methods import Optimizer
from ..acquisition_scheme import AcquisitionScheme
from ..tissue_model import TissueModel
from ..utils.IO import initiate_logging_directory

# Set up the logger
log_dir = initiate_logging_directory()

current_time = datetime.now().strftime('%y%m%d_%H%M')
log_filename = f"optimization_{current_time}.log"
logging.basicConfig(filename=os.path.join(log_dir, log_filename),
                    level=logging.INFO,
                    format='%(message)s')

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
    :param solver_options: Options specific to the solver check the documentation of the solver to see what can be done.
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
    acquisition_parameter_scales = scheme_copy.free_parameter_scales
    x0 = scheme_copy.free_parameter_vector / acquisition_parameter_scales
    scaled_bounds = scheme_copy.free_parameter_bounds_scaled
    scipy_bounds = bounds_tuple2scipy(scaled_bounds)
    constraints = scheme_copy.constraints

    # The parameters required to fully define the loss function.
    scipy_loss_args = (scheme_copy, model, noise_variance, loss, loss_scaling_factor)

    if method == 'differential_evolution':
        # constraints are formatted different for this optimization method
        if constraints is None:
            constraints = ()

        result = differential_evolution(scipy_loss, bounds=scipy_bounds,
                                        args=scipy_loss_args,
                                        x0=x0, workers=-1, disp=True, updating='deferred', constraints=constraints,
                                        polish=True, callback=progress_callback_DE, **solver_options)
    else:
        callback = make_local_callback(scheme_copy, model, noise_variance)
        result = minimize(scipy_loss, x0, args=scipy_loss_args,
                          method=method, bounds=scipy_bounds, constraints=constraints,
                          options=solver_options, callback=callback)

    # update the scheme_copy to the result found by the optimizer
    if 'x' in result:
        scheme_copy.set_free_parameter_vector(result['x'] * acquisition_parameter_scales)

    # check if the optimized scheme is better than the initial scheme
    current_loss = compute_loss(scheme_copy, model, noise_variance, loss)
    if current_loss > initial_loss:
        raise RuntimeError("Loss increased during optimization, try a different optimization method.")

    warn_early_termination(result)

    return scheme_copy, result


def log_callback(iteration, parameters, objective_function, other_stuff: Dict[str, str] = None):
    logging.info("Iteration %d:", iteration)
    logging.info("\t Parameters: %s", parameters)
    logging.info("\t Objective Function Value: %f", objective_function)
    if other_stuff is not None:
        for key, val in other_stuff.items():
            logging.info(f"\t {key}: {val}")
    logging.info("------------------------------")


def make_local_callback(scheme: AcquisitionScheme, model: TissueModel, noise_var: float) -> callable:
    def callback(x_current, intermediate_result: OptimizeResult):
        fun = intermediate_result.fun
        iteration = intermediate_result.nit
        jac = intermediate_result.jac

        scheme.set_free_parameter_vector(x_current * scheme.free_parameter_scales)
        crlbs = compute_crlbs(scheme, model, noise_var)
        log_callback(iteration, x_current, fun, other_stuff={"Scaled CRLBs": f"{crlbs}", "Jacobian": f"{jac}"})

    return callback


def progress_callback_DE(x_current, convergence):
    print("I am being called")

    logging.info(f"Current best solution: {x_current} | OF: NotImplemented | Convergence {convergence}")


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

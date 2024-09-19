import logging
import os
import warnings
from copy import deepcopy
from datetime import datetime
from typing import Optional, Union, Tuple, List, Dict

import numpy as np
from scipy.optimize import OptimizeResult, minimize, differential_evolution, Bounds

from .loss_functions import compute_loss, scipy_loss, LossFunction, default_loss, ILL_LOSS, compute_crlb
from .methods import Optimizer
from ..acquisition_scheme import AcquisitionScheme
from ..constants import ConstraintTypes
from ..tissue_model import TissueModel, RelaxationTissueModel
from ..utils.IO import initiate_logging_directory

# Set up the logger
log_dir = initiate_logging_directory()

current_time = datetime.now().strftime('%y%m%d_%H%M')
log_filename = f"optimization_{current_time}.log"
logging.basicConfig(filename=os.path.join(log_dir, log_filename),
                    level=logging.INFO,
                    format='%(message)s')


def iterative_shell_optimization(
        scheme: AcquisitionScheme,
        model: RelaxationTissueModel,
        n_shells: int,
        n_directions: int,
        iterations: int,
        noise_variance: float,
        loss: LossFunction = default_loss,
        loss_scaling_factor: float = 1.0,
        method: Optional[Union[str, Optimizer]] = "differential_evolution",
        solver_options: dict = None) -> AcquisitionScheme:
    """
    Iteratively optimizes the free parameters in the acquisition scheme and returns the scheme with the lowest loss.
    The b-values and gradient directions are randomly initialized on the given number of shells and directions per
    shell after each iteration.
    """

    optimal_loss = None
    optimal_scheme = scheme

    for i in range(iterations):
        scheme.fix_b0_measurements()
        
        print(f"Starting iteration {i}")
        optimized_scheme, optimized_loss = optimize_scheme(
            scheme,
            model,
            noise_variance=noise_variance,
            loss=loss,
            loss_scaling_factor=loss_scaling_factor,
            method=method,
            solver_options=solver_options
        )
        if i == 0 or optimized_loss < optimal_loss:
            optimal_scheme = optimized_scheme
            optimal_loss = optimized_loss
        
        print(f"Finished iteration {i}")

        model_dependencies = model.get_dependencies()
        scheme = scheme.random_shell_initialization(n_shells, n_directions, model_dependencies)
    
    return optimal_scheme


def optimize_scheme(scheme: AcquisitionScheme, model: TissueModel,
                    noise_variance: float,
                    loss: LossFunction = default_loss,
                    loss_scaling_factor: float = 1.0,
                    method: Optional[Union[str, Optimizer]] = "differential_evolution",
                    solver_options: dict = None) -> Tuple[AcquisitionScheme, float]:
    """
    Optimizes the free parameters in the given MR acquisition scheme such that the loss is minimized.
    The loss function should be of type LossFunction, which takes an N×M Jacobian matrix, an array with M parameter
    scales, and the noise variance. The loss function should return a scalar loss. N is the number of measurements
    in the acquisition and M is the number of tissue parameters.

    :param scheme: The MR acquisition scheme to be optimized. (NOTE: a reference of the scheme is passed,
                    so it will be changed).
    :param model: The tissue model for which we want the optimal acquisition scheme.
    :param noise_variance: Noise variance on the MR signal attenuation.
    :param loss: a function of type LossFunction.
    :param loss_scaling_factor: Can be used to scale the loss function to order 1.0 if you notice extreme values.
    :param method: Either 'differential_evolution' or any solver available in scipy.optimize.minimize that does not
                    require a Jacobian.
    :param solver_options: Options specific to the solver, provided as a dictionary with the options as keywords, see
                            the documentation of the solver for the keywords.
    :return: The optimized acquisition scheme and the loss.
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
    x0 = scheme_copy.scaled_free_parameter_vector

    scaled_bounds = scheme_copy.free_parameter_bounds_scaled
    scipy_bounds = bounds_tuple2scipy(scaled_bounds)
    constraints = scheme_copy.constraint_list
    check_constraints_satisfied(x0, scheme_copy.constraints)
    model.check_dependencies(scheme_copy)
    
    # The parameters required to fully define the loss function.
    scipy_loss_args = (scheme_copy, model, noise_variance, loss, loss_scaling_factor)

    if method == 'differential_evolution':
        # constraints are formatted different for this optimization method
        if constraints is None:
            constraints = ()

        callback = make_de_callback(scheme_copy, model, noise_variance, loss)

        logging.info("Optimizing with differential evolution optimizer.")
        result = differential_evolution(scipy_loss, bounds=scipy_bounds,
                                        args=scipy_loss_args,
                                        x0=x0, workers=-1, disp=True, updating='deferred', constraints=constraints,
                                        polish=True, callback=callback, **solver_options)
    else:
        logging.info(f"Optimizing with {method}.")
        callback = make_local_callback(scheme_copy, model, noise_variance, loss)
        result = minimize(scipy_loss, x0, args=scipy_loss_args,
                          method=method, bounds=scipy_bounds, constraints=constraints,
                          options=solver_options, callback=callback)

    # update the scheme_copy to the result found by the optimizer
    if 'x' in result:
        x = result['x']
        scheme_copy.set_scaled_free_parameter_vector(result['x'])
    else:
        raise RuntimeError("No suitable solution was found?")

    # check if the optimized scheme is better than the initial scheme
    current_loss = compute_loss(scheme_copy, model, noise_variance, loss)
    if current_loss > initial_loss:
        Warning("Loss increased during optimization, try a different optimization method.")

    check_constraints_satisfied(x, scheme_copy.constraints)

    warn_early_termination(result)

    return scheme_copy, current_loss


def log_callback(iteration: int, parameters: np.ndarray, objective_function: float, more_info: Dict[str, str] = None):
    """
    This function is for formatting the log output of callback functions

    :param iteration: The iteration count
    :param parameters: The current best solution the optimizer found
    :param objective_function: The value of the objective function for these parameters
    :param more_info: A dictionary containing more info to print to the log
    """
    logging.info("Iteration %d:", iteration)
    logging.info("\t Parameters: %s", parameters)
    logging.info("\t Objective Function Value: %f", objective_function)
    if more_info is not None:
        for key, val in more_info.items():
            logging.info(f"\t {key}: {val}")
    logging.info("------------------------------")


def make_local_callback(scheme: AcquisitionScheme,
                        model: TissueModel,
                        noise_var: float,
                        loss: LossFunction) -> callable:
    """
    A maker function for the callback currently only tested with trust-constr method.

    :param scheme: Acquisition scheme used in optimization
    :param model: Tissue model used in optimization
    :param noise_var: The chosen noise variance
    :param loss: The used loss function
    :return: A callback function for use with scipy.optimize.minimize methods that support intermediate results
    """
    # Using mutable object to track iterations
    iteration_tracker = [0]

    def callback(x_current, intermediate_result: Optional[OptimizeResult] = None, *_args, **_kwargs):
        if intermediate_result is None:
            iteration_tracker[0] += 1
            fun = compute_loss(scheme, model, noise_var, loss)
            crlb = compute_crlb(scheme, model, noise_var, loss)
            log_callback(iteration_tracker[0], x_current, fun, more_info={"Scaled CRLBs": f"{crlb}"})
        else:
            fun = intermediate_result.fun
            iteration = intermediate_result.nit
            jac = intermediate_result.jac

            scheme.set_scaled_free_parameter_vector(x_current)
            crlb = compute_crlb(scheme, model, noise_var, loss)
            log_callback(iteration, x_current, fun, more_info={"Scaled CRLBs": f"{crlb}", "Jacobian": f"{jac}"})

    return callback


def make_de_callback(scheme: AcquisitionScheme, model: TissueModel, noise_var: float, loss: LossFunction) -> callable:
    """
    A maker function for the callback function used with differential evolution optimizer.
    Might be applicable to other methods but used here only with differential evolution.

    :param scheme: Acquisition scheme used in optimization
    :param model: Tissue model used in optimization
    :param noise_var: The chosen noise variance
    :param loss: The used loss function
    :return: a callback function for differential evolution optimization method.
    """
    # Using mutable object to track iterations
    iteration_tracker = [0]

    def callback(x_current, *_args, **_kwargs):
        iteration_tracker[0] += 1
        scheme.set_scaled_free_parameter_vector(x_current)
        fun = compute_loss(scheme, model, noise_var, loss)
        crlb = compute_crlb(scheme, model, noise_var, loss)
        log_callback(iteration_tracker[0], x_current, fun, {"Scaled CRLBs": f"{crlb}"})

    return callback


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
            "Possible solutions include but are not limited to: Changing the optimizer settings, changing the "
            "optimization method or changing the initial scheme to a more suitable one. "
            "If you are using a scipy optimizer its settings can be changed by passing options to this function. "
            "If you are using a MICROtool optimization method please consult the optimization_methods module for more "
            "details.")


def check_ill_conditioned(loss_value: float):
    if loss_value == ILL_LOSS:
        raise RuntimeError(
            "Initial AcquisitionScheme error: the initial acquisition scheme results in an ill conditioned Fisher "
            "information matrix, possibly due to model degeneracy. "
            "Try a different initial AcquisitionScheme, or alternatively simplify your TissueModel.")


def check_insensitive(scheme: AcquisitionScheme, model: TissueModel):
    jac = model.scaled_jacobian(scheme)
    # the parameters we included for optimization but to which the signal is insensitive
    # (jac is signal derivative for all parameters)
    insensitive_parameters = np.all(jac == 0, axis=0)
    if np.any(insensitive_parameters):
        params = np.array(model.parameter_names)[model.include_optimize][insensitive_parameters]
        raise ValueError(
            f"Initial AcquisitionScheme error: the parameters {params} have a zero signal derivative for all "
            "measurements. Optimizing will not result in a scheme that better estimates these parameters. "
            "Exclude them from optimization if you are okay with that.")


def check_degrees_of_freedom(scheme: AcquisitionScheme, model: TissueModel):
    m = int(np.sum(np.array(model.include_optimize)))
    n = scheme.pulse_count
    if m > n:
        raise ValueError(f"The TissueModel has too many degrees of freedom ({m}) to optimize the "
                         f"AcquisitionScheme parameters ({n}) with meaningful result.")


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
            error_msg += f"The {key} constraint is not satisfied. With violation {value} not in " \
                         f"[{constraint.lb},{constraint.ub}]"

    if error_msg != "":
        raise RuntimeError(error_msg)

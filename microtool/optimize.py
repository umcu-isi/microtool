from copy import deepcopy
from typing import Optional, Union, Tuple, TypeVar

import numpy as np
from scipy.optimize import OptimizeResult, minimize, differential_evolution, dual_annealing

from microtool.acquisition_scheme import AcquisitionScheme
from microtool.loss_function import compute_loss, compute_loss_init, scipy_loss, LossFunction, default_loss
from microtool.optimization_methods import Optimizer
from microtool.tissue_model import TissueModel

# Arbitrary high cost value for ill conditioned matrices
ILL_COST = 1e9

# A way of type hinting all the derived classes of AcquisitionScheme
AcquisitionType = TypeVar('AcquisitionType', bound=AcquisitionScheme)


def optimize_scheme(scheme: AcquisitionType, model: TissueModel,
                    noise_variance: float,
                    loss: LossFunction = default_loss,
                    method: Optional[Union[str, Optimizer]] = None,
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
    :return: A scipy.optimize.OptimizeResult object.
    """
    M = int(np.sum(np.array(model.include)))

    # Checking if optimization options are provided.
    if 'options' in kwargs.keys():
        optimizer_options = kwargs['options']
    else:
        optimizer_options = None

    # the loss of the initial scheme
    initial_loss = np.array([compute_loss_init(scheme, model, noise_variance, loss=loss)])

    # Copying the scheme for number repeated optimizations required.
    scheme_copy = deepcopy(scheme)

    N = len(scheme.free_parameter_vector)
    if M > N:
        raise ValueError(f"The TissueModel has too many degrees of freedom ({M}) to optimize the "
                         f"AcquisitionScheme parameters ({N}) with meaningful result.")

    acquisition_parameter_scales = scheme_copy.free_parameter_scales
    x0 = scheme_copy.free_parameter_vector / acquisition_parameter_scales
    scaled_bounds = scheme_copy.free_parameter_bounds_scaled
    constraints = scheme_copy.get_constraints()

    # The parameters required to fully define the loss function.
    scipy_loss_args = (scheme_copy, model, noise_variance, loss)

    if method == 'differential_evolution':
        # converting dictionary constraints of microtool tissuemodel to scipy NonLinear constraint
        # constraints = scipy.optimize.NonlinearConstraint(constraints['fun'],0,np.inf)
        result = differential_evolution(scipy_loss, bounds=scaled_bounds,
                                        args=scipy_loss_args,
                                        x0=x0, workers=-1, disp=True, updating='deferred')
    elif method == 'dual_annealing':
        result = dual_annealing(scipy_loss, scaled_bounds, scipy_loss_args, x0=x0)
    else:
        result = minimize(scipy_loss, x0, args=scipy_loss_args,
                          method=method, bounds=scaled_bounds, constraints=constraints,
                          options=optimizer_options)

    if 'x' in result:
        scheme_copy.set_free_parameter_vector(result['x'] * acquisition_parameter_scales)

    # check if the optimized scheme is better than the initial scheme and update
    # also save best optimization result.
    current_loss = compute_loss(scheme_copy, model, noise_variance, loss=loss)
    if current_loss > initial_loss:
        raise RuntimeError("Loss increased during optimization, try a different optimization method.")

    optimized_scheme = scheme_copy
    optimize_result = result

    if not optimize_result["success"]:
        print(result)
        raise RuntimeError(
            "Optimization procedure was unsuccesfull. "
            "Possible solutions include but are not limited to: Changing the "
            "optimizer setings, changing the optimization method or changing the initial scheme to a more suitable one."
            "If you are using a scipy optimizer its settings can be changed by passing options to this function. "
            "If you are using a microtool optimization method please consult the optimization_methods module for more details.")

    return optimized_scheme, optimize_result

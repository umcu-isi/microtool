from typing import List, Tuple, Optional

import numpy as np
from scipy.optimize import LinearConstraint


def is_constrained(x: np.ndarray, constraints: LinearConstraint) -> bool:
    """A function for checking if a given parameter combination breaks a given linear constraint.

    :param x: Parameter combination
    :param constraints: scipy linear constraint object
    :return: boolean that is true if the parameter combination breaks the constraint
    """
    # Readability variables
    A = constraints.A
    lb = constraints.lb
    ub = constraints.ub

    transformed_parameters = A @ x
    return np.any((lb >= transformed_parameters) | (transformed_parameters >= ub))


def check_bounded(allbounds: List[Tuple[Optional[float], Optional[float]]]) -> None:
    """This function checks the boundedness of a set of given bounds such that brute force optimizers
    can assume boundedness after calling this function.

    :param allbounds: A list of bounds
    :raises ValueError: Raises a value error in case the there are no bounds or if bounds are to large
    """
    # Check for finite boundaries
    if allbounds is None:
        raise ValueError(
            " No bounds provided in optimize: this optimization can only be executed on a finite domain")
    for bounds in allbounds:
        for bound in bounds:
            if bound is None:
                raise ValueError(" Infinite boundaries not supported for this optimizer")
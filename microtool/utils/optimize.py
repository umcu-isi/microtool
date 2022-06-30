from typing import List, Tuple, Optional

import numpy as np
from tqdm.contrib.itertools import product


def is_constrained(x: np.ndarray, A: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> bool:
    """A function for checking if a given parameter combination breaks a given linear constraint.

    :param A:
    :param lb:
    :param ub:
    :param x: Parameter combination
    :return: boolean that is true if the parameter combination breaks the constraint
    """
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


def find_minimum(fun: callable, domains: List[np.ndarray], constraints) -> Tuple[np.ndarray, float]:
    if constraints != ():
        A = constraints.A
        lb = constraints.lb
        ub = constraints.ub

        # iterate over the grid
        y_optimal = np.inf
        x_optimal = None
        for combination in product(*domains, desc='Running brute force grid computation'):
            combination = np.array(combination)
            # check constraint
            if is_constrained(combination, A, lb, ub):
                loss = np.inf
            else:
                loss = fun(combination)

            # update optimal value
            if loss < y_optimal:
                x_optimal = combination
                y_optimal = loss
    else:
        # iterate over the grid
        y_optimal = np.inf
        x_optimal = None
        for combination in product(*domains, desc='Running brute force grid computation'):
            combination = np.array(combination)
            loss = fun(combination)
            # update optimal value
            if loss < y_optimal:
                x_optimal = combination
                y_optimal = loss

    return x_optimal, y_optimal

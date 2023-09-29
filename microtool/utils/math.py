from typing import Union

import numpy as np

Number = Union[np.ndarray, float]


def is_smaller_than_with_tolerance(number: Number, lower_bound: Number, tolerance=1e-17) -> Union[bool, np.ndarray]:
    """
    This function returns True if number is strictly smaller than a lower bound where we include a tolerance for which
    we consider number to be equal to lower bound and return False.

    :param number: The number which we wish to check
    :param lower_bound: The lower bound
    :param tolerance: The tolerance that we allow for equality with the lower bound
    :return: True if strictly lower than lower bound with a given tolerance. False otherwise
    """
    # array with values close to lb true
    close_to_lb = np.isclose(number, lower_bound, atol=tolerance)

    # normal comparison array
    less_than_lb = number < lower_bound

    # set values within tolerance to False (we consider this equality with the bound)
    less_than_lb[close_to_lb] = False
    return less_than_lb


def is_higher_than_with_tolerance(number: Number, upper_bound: Number, tolerance=1e-17) -> Union[bool, np.ndarray]:
    """
    This function returns True if number is strictly larger than a lower bound where we include a tolerance for which
    we consider number to be equal to upperbound and return False.

    :param number: The number which we wish to check
    :param upper_bound: The upper bound
    :param tolerance: The tolerance that we allow for equality with the lower bound
    :return: True if strictly higher than higher with a given tolerance. False otherwise
    """
    # Using math we reuse the function above
    return is_smaller_than_with_tolerance(-1. * number, -1. * upper_bound, tolerance)

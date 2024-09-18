import numpy as np

from microtool.utils.math import largest_real_cbrt
from microtool.utils.unit_registry import unit


def test_largest_real_cbrt():
    # These cubics (x³ + a2 x² + a1 x + a2 = 0) all have solutions including x = 2 being the largest.
    cubics = np.array([[0, 0, -8], [0, -4, 0], [-2, 0, 0], [0, -2, -4], [-1, 0, -4], [-1, -2, 0]])
    roots = largest_real_cbrt(
        cubics[:, 0] * unit('s'),
        cubics[:, 1] * unit('s²'),
        cubics[:, 2] * unit('s³'))
    assert np.allclose(roots, 2 * unit('s'))

    # This cubic does not have a solution and should therefore return NaN.
    roots = largest_real_cbrt(1, 1, 1)
    assert np.isnan(roots)

import numpy as np

from microtool.utils.math import largest_real_cbrt, real_cbrt, newton_polynomial_root
from microtool.utils.unit_registry import unit


def test_largest_real_cbrt():
    # These cubics (x³ + a2 x² + a1 x + a0 = 0) all have solutions including x = 2 being the largest.
    cubics = np.array([[0, 0, -8], [0, -4, 0], [-2, 0, 0], [0, -2, -4], [-1, 0, -4], [-1, -2, 0]])
    roots = largest_real_cbrt(
        cubics[:, 0] * unit('s'),
        cubics[:, 1] * unit('s²'),
        cubics[:, 2] * unit('s³'))
    assert np.allclose(roots, 2 * unit('s'), rtol=1e-9)

    # This cubic does not have a solution and should therefore return NaN.
    roots = largest_real_cbrt(1, 1, 1)
    assert np.isnan(roots)


def test_real_cbrt():
    # These cubics (x³ + a2 x² + a1 x + a0 = 0) all have one real root x=2.
    cubics = np.array([[0, 0, -8], [0, -2, -4], [-1, 0, -4]])
    roots = real_cbrt(
        cubics[:, 0] * unit('s'),
        cubics[:, 1] * unit('s²'),
        cubics[:, 2] * unit('s³'))
    assert np.allclose(roots, 2 * unit('s'), rtol=1e-9)

    # This cubic does not have a solution and should therefore return NaN.
    roots = real_cbrt(1, 1, 1)
    assert np.isnan(roots)


def test_newton_polynomial_root():
    # Start on different locations.
    for x0 in np.array([[1, 1, 1], [3, 3, 3]], dtype=float) * unit('s'):
        # These cubics (c3 x³ + c2 x² + c1 x + c0 = 0) all have one real root x=2.
        cubics = np.array([[0, 0, -8], [0, -2, -4], [-1, 0, -4]])
        roots = newton_polynomial_root([
            cubics[:, 2] * unit('s³'),
            cubics[:, 1] * unit('s²'),
            cubics[:, 0] * unit('s'),
            1 * unit('')], x0, n=10)
        assert np.allclose(roots, 2 * unit('s'), rtol=1e-9)

    # Test x³ = 0, starting at the solution x = 0. This is an extreme case because it's derivative is zero at that
    # point.
    assert newton_polynomial_root([None, None, None, 1], 0, n=10) == 0

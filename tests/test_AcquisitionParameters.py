import numpy as np

from microtool.acquisition_scheme import AcquisitionParameters


def test_repeated_parameters():
    """
    We should be able to have repeated parameters over only the free parameteres, we test this by prepending a fixed
    parameter to the start of this parameter.
    """
    N = 4  # number of measurement without the prepended fixed measurement
    P = 2  # period of the repeated measurement
    vals = np.repeat(1.0, N)
    vals = np.insert(vals, 0, 0)
    parameter = AcquisitionParameters(values=vals, unit='s', scale=1.0, symbol="T")

    # fixing the first measurement (not to be included in optimization)
    fixed = np.zeros(vals.shape)
    fixed[0] = 1
    parameter.set_fixed_mask(fixed)

    # testing the setter
    assert parameter._repetition_period == 0
    parameter.set_repetition_period(P)
    assert parameter._repetition_period == P

    # testing the update
    parameter.free_values = np.array([0.5, 0.25])
    expected_before_update = np.array([0, 0.5, 1, 0.25, 1])
    np.testing.assert_allclose(parameter.values, expected_before_update)

    parameter.update_repeated_values()

    # after updateing repeated values we expect
    expected_value_array = np.array([0, 0.5, 0.5, 0.25, 0.25])
    np.testing.assert_allclose(parameter.values, expected_value_array)

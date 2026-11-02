import numpy as np

from microtool.acquisition_scheme import AcquisitionParameters


def test_repeated_parameters():
    """
    We should be able to have repeated parameters over only the free parameters, we test this by prepending a fixed
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

    # after updating repeated values we expect
    expected_value_array = np.array([0, 0.5, 0.5, 0.25, 0.25])
    np.testing.assert_allclose(parameter.values, expected_value_array)


def test_repeated_parameters_with_interleaved_fixed_measurements():
    n_shells = 5
    n_directions = 6
    block_size = n_directions + 1
    values = np.repeat(np.arange(n_shells, dtype=float), block_size)
    b0_mask = np.arange(values.size) % block_size == 0
    values[b0_mask] = 0.0
    parameter = AcquisitionParameters(values=values, unit='mT/mm', scale=1.0)

    parameter.set_fixed_mask(b0_mask)
    parameter.set_repetition_period(n_directions)

    expected_free_indices = np.arange(n_shells) * block_size + 1
    np.testing.assert_array_equal(
        np.flatnonzero(parameter.optimize_mask), expected_free_indices
    )

    parameter.free_values = np.arange(10.0, 10.0 + n_shells)
    parameter.update_repeated_values()

    expected_values = np.repeat(np.arange(10.0, 10.0 + n_shells), block_size)
    expected_values[b0_mask] = 0.0
    np.testing.assert_allclose(parameter.values, expected_values)


class TestAcquisitionParameters:
    parameter = AcquisitionParameters(
        values=np.array([1., 2., 3.]), unit='m', scale=1.0
    )

    expected_fix_mask = np.array([True, False, False])

    def test_set_fixed_mask(self):
        """
        Testing if fixed mask number of truth values matches free parameter length
        """
        # fix only first measurement

        self.parameter.set_fixed_mask(self.expected_fix_mask)

        # testing if the optimize mask matches input fixed mask
        actual_fix_mask = np.logical_not(self.parameter.optimize_mask)
        assert np.all(self.expected_fix_mask == actual_fix_mask)

        # testing if the number of free parameters matches the mask we provided
        expected_free_parameter_length = np.sum(np.logical_not(self.expected_fix_mask))
        actual_free_parameter_length = len(self.parameter.free_values)
        assert expected_free_parameter_length == actual_free_parameter_length

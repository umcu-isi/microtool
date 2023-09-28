import numpy as np

from microtool.acquisition_scheme import AcquisitionParameters


class TestAquisitionParameters:
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

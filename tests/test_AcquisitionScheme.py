import numpy as np
import pytest

from microtool.acquisition_scheme import AcquisitionScheme, AcquisitionParameters
from microtool.constants import ConstraintTypes


class SimpleScheme(AcquisitionScheme):
    def __init__(self, T):
        super().__init__({
            "Time": AcquisitionParameters(
                values=T, unit="s", scale=1.0
            )
        })

    @property
    def constraints(self) -> ConstraintTypes:
        return None


class TestAcquisitionScheme:
    scheme = SimpleScheme(np.array([1., 2., 3.]))
    # so we have only first measurement free
    scheme["Time"].set_fixed_mask(np.array([True, False, False]))

    def test_set_free_parameter_vector_error_catching(self):
        new_free_parameter_values = np.array([0.5, 0.6, 0.7])

        # using too many values should raise error
        with pytest.raises(ValueError):
            self.scheme.set_free_parameter_vector(new_free_parameter_values)

    def test_set_free_parameter_vector(self):
        new_free_parameter_values = np.array([0.6, 0.7])
        expected_parameter_vector = np.array([1., 0.6, 0.7])

        self.scheme.set_free_parameter_vector(new_free_parameter_values)
        np.testing.assert_equal(self.scheme.free_parameter_vector, new_free_parameter_values)

        np.testing.assert_equal(self.scheme["Time"].values, expected_parameter_vector)

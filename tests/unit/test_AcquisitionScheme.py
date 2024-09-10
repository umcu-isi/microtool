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
        return []


class LessSimpleScheme(AcquisitionScheme):
    def __init__(self, t, s):
        super().__init__({
            "Time": AcquisitionParameters(
                values=t, unit="s", scale=1.0
            ),
            "Space": AcquisitionParameters(
                values=s, unit="m", scale=0.1, lower_bound=0, upper_bound=10,
            )
        })

    @property
    def constraints(self) -> ConstraintTypes:
        return []


class TestLessSimpleScheme:
    scheme = LessSimpleScheme(np.array([1, 2, 3]), np.array([4, 5, 6]))
    scheme["Time"].fixed = True
    scheme["Space"].set_fixed_mask(np.array([True, False, False]))

    def test_set_free_parameter_vector(self):
        new_free_vector = np.array([7, 8])

        expected_values_space = np.array([4, 7, 8])

        self.scheme.set_free_parameter_vector(new_free_vector)
        np.testing.assert_equal(self.scheme.free_parameter_vector, new_free_vector)

        np.testing.assert_equal(self.scheme["Space"].values, expected_values_space)

    def test_bounds(self):
        # There are two free parameters. Their bounds are (0, 10) and parameter scale 0.1.
        expected_bounds = [(0, 100), (0, 100)]
        np.testing.assert_equal(self.scheme.free_parameter_bounds_scaled, expected_bounds)

        # Creating a scheme with out-of-bound values should raise an error.
        with pytest.raises(ValueError):
            LessSimpleScheme(np.array([1]), np.array([10.1]))  # 10.1 > 10
        with pytest.raises(ValueError):
            LessSimpleScheme(np.array([1]), np.array([-0.1]))  # -0.1 < 0


class TestSimpleAcquisitionScheme:
    scheme = SimpleScheme(np.array([1., 2., 3.]))
    # so we have only first measurement free
    scheme["Time"].set_fixed_mask(np.array([True, False, False]))

    def test_set_free_parameter_vector(self):
        new_free_parameter_values = np.array([0.6, 0.7])
        expected_parameter_vector = np.array([1., 0.6, 0.7])

        self.scheme.set_free_parameter_vector(new_free_parameter_values)
        np.testing.assert_equal(self.scheme.free_parameter_vector, new_free_parameter_values)

        np.testing.assert_equal(self.scheme["Time"].values, expected_parameter_vector)

        # using too many values should raise error
        new_free_parameter_values = np.array([0.5, 0.6, 0.7])
        with pytest.raises(ValueError):
            self.scheme.set_free_parameter_vector(new_free_parameter_values)

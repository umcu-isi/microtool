"""
I need to test that the pulse relations implemented actually make sense
"""
import pytest

from microtool.constants import GAMMA, B_UNIT, GRADIENT_UNIT, PULSE_TIMING_UNIT, GAMMA_UNIT
from microtool.pulse_relations import get_b_value_simplified
from unit_registry import Q_


def test_simple_pulse_relation():
    """
    Testing if the parameter set from alexander 2008 can be reproduced
    :return:
    """
    expected_b = Q_(20087, B_UNIT)

    # other pulse parameters
    Delta = Q_(.025, PULSE_TIMING_UNIT)
    delta = Q_(.02, PULSE_TIMING_UNIT)
    G_magnitude = Q_(.2, GRADIENT_UNIT)
    gamma = Q_(GAMMA, GAMMA_UNIT)
    computed_b = get_b_value_simplified(gamma, G_magnitude, Delta, delta)

    assert computed_b.units == expected_b.units
    assert computed_b.magnitude == pytest.approx(expected_b.magnitude, abs=1e3)

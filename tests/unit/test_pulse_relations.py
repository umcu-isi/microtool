"""
I need to test that the pulse relations implemented actually make sense
"""
import pytest

from microtool.constants import *
from microtool.pulse_relations import get_b_value_simplified, get_b_value_complete, compute_t_rise, get_gradients
from test_ScannerParameters import scanner_parameters
from unit_registry import Q_, ureg, gamma_wunits

expected_b = Q_(20087, B_UNIT)

# other pulse parameters
Delta = Q_(.025, PULSE_TIMING_UNIT)
delta = Q_(.02, PULSE_TIMING_UNIT)
G_magnitude = Q_(.2, GRADIENT_UNIT)


def test_simple_pulse_relation():
    """
    Testing if the parameter set from alexander 2008 can be reproduced
    :return:
    """

    computed_b = get_b_value_simplified(gamma_wunits, G_magnitude, Delta, delta)

    assert computed_b.units == expected_b.units
    assert computed_b.magnitude == pytest.approx(expected_b.magnitude, abs=1e3)


class TestFullPulseRelation:
    def test_t_rise(self):
        t_r = compute_t_rise(G_magnitude, scanner_parameters)

        # since we convert units in this particular function we add a correction factor
        t_r = t_r.to('s') * 1e3

        assert t_r.units == ureg.second

    def test_b_value(self):
        computed_b = get_b_value_complete(gamma_wunits, G_magnitude, Delta, delta, scanner_parameters)
        assert computed_b.units == expected_b.units
        assert computed_b.magnitude == pytest.approx(expected_b.magnitude, abs=1e3)


def test_get_gradients():
    predicted_G = get_gradients(gamma_wunits, expected_b, Delta, delta, scanner_parameters)
    assert predicted_G.units == G_magnitude.units
    assert predicted_G.magnitude == pytest.approx(G_magnitude.magnitude, rel=.1)

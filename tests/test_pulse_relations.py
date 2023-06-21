"""
I need to test that the pulse relations implemented actually make sense
"""
import pytest

from microtool.constants import *
from microtool.pulse_relations import get_b_value_simplified, get_b_value_complete, compute_t_rise
from microtool.scanner_parameters import ScannerParameters
from unit_registry import Q_, ureg

expected_b = Q_(20087, B_UNIT)

# other pulse parameters
Delta = Q_(.025, PULSE_TIMING_UNIT)
delta = Q_(.02, PULSE_TIMING_UNIT)
G_magnitude = Q_(.2, GRADIENT_UNIT)
gamma = Q_(GAMMA, GAMMA_UNIT)

# Typical scanner parameters
t90 = Q_(4.e-3, PULSE_TIMING_UNIT)
t_180 = Q_(6.e-3, PULSE_TIMING_UNIT)
t_half = Q_(14e-3, PULSE_TIMING_UNIT)
g_max = Q_(200e-3, GRADIENT_UNIT)
s_max = Q_(1300., SLEW_RATE_UNIT)
scanner_parameters = ScannerParameters(t90, t_180, t_half, g_max, s_max)


def test_simple_pulse_relation():
    """
    Testing if the parameter set from alexander 2008 can be reproduced
    :return:
    """

    computed_b = get_b_value_simplified(gamma, G_magnitude, Delta, delta)

    assert computed_b.units == expected_b.units
    assert computed_b.magnitude == pytest.approx(expected_b.magnitude, abs=1e3)


class TestFullPulseRelation:
    def test_t_rise(self):
        t_r = compute_t_rise(G_magnitude, scanner_parameters)
        assert t_r.units == ureg.second

    def test_b_value(self):
        computed_b = get_b_value_complete(gamma, G_magnitude, Delta, delta, scanner_parameters)
        assert computed_b.units == expected_b.units
        assert computed_b.magnitude == pytest.approx(expected_b.magnitude, abs=1e3)

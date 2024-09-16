"""
I need to test that the pulse relations implemented actually make sense
"""
import pytest

from microtool.scanner_parameters import ScannerParameters
from microtool.pulse_relations import compute_b_values, compute_t_rise, get_gradients  # noqa: E402
from microtool.utils.unit_registry import unit


t_90 = 4e-3 * unit('s')
t_180 = 6e-3 * unit('s')
t_half = 14e-3 * unit('s')
g_max = 200e-3 * unit('mT/mm')  # typical is 40 mT/m = 40e-3 mT/mm
s_max = 1300 * unit('mT/mm/s')  # typical is 1300 T/m/s = 1300 mT/mm/s
scanner_parameters = ScannerParameters(t_90, t_180, t_half, g_max, s_max)

expected_t_rise = 0.153846e-3 * unit('s')
expected_b = 20.9933e3 * unit('s/mm²')  # TODO: why was it 20087 before? Where did that number come from?
# expected_b = 20087 * unit('s/mm²')

# other pulse parameters
pulse_interval = 25e-3 * unit('s')
pulse_width = 20e-3 * unit('s')
g_magnitude = 200e-3 * unit('mT/mm')


def test_simple_pulse_relation():
    """
    Testing if the parameter set from alexander 2008 can be reproduced
    :return:
    """

    computed_b = compute_b_values(g_magnitude, pulse_interval, pulse_width)

    assert computed_b.units == expected_b.units
    assert computed_b.magnitude == pytest.approx(expected_b.magnitude, rel=1e-5)


class TestFullPulseRelation:
    def test_t_rise(self):
        t_r = compute_t_rise(g_magnitude, scanner_parameters)
        assert t_r.units == expected_t_rise.units
        assert t_r.magnitude == pytest.approx(expected_t_rise.magnitude, rel=1e-5)

    def test_b_value(self):
        b = compute_b_values(g_magnitude, pulse_interval, pulse_width, scanner_parameters=scanner_parameters)
        assert b.units == expected_b.units
        assert b.magnitude == pytest.approx(expected_b.magnitude, rel=1e-5)

        # Test approximate b-values (without scanner parameters).
        # TODO: is the difference between the full version and the approximation indeed that small?
        b = compute_b_values(g_magnitude, pulse_interval, pulse_width)
        assert b.units == expected_b.units
        assert b.magnitude == pytest.approx(expected_b.magnitude, rel=1e-5)


def test_get_gradients():
    g = get_gradients(expected_b, pulse_interval, pulse_width, scanner_parameters)
    assert g.units == g_magnitude.units
    assert g.magnitude == pytest.approx(g_magnitude.magnitude, rel=1e-5)

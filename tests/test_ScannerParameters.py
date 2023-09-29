from microtool.constants import GRADIENT_UNIT, SLEW_RATE_UNIT
from microtool.scanner_parameters import ScannerParameters
from unit_registry import Q_, ureg

t90 = Q_(4.e-3, 's')
t_180 = Q_(6.e-3, 's')
t_half = Q_(14e-3, 's')

# typical is 40 mT/m = 40e-3 mT/mm
g_max = Q_(200e-3, GRADIENT_UNIT)

# typical is 1300 T/m/s = 1300 mT/mm/s
s_max = Q_(1300., SLEW_RATE_UNIT)
scanner_parameters = ScannerParameters(t90, t_180, t_half, g_max, s_max)


def test_inferred_units():
    """
    Making sure that correct units are inferred
    """
    assert scanner_parameters.t_rise.units == ureg('s').units

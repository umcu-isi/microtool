from microtool import ureg
from microtool.scanner_parameters import default_scanner


def test_inferred_units():
    """
    Making sure that correct units are inferred
    """
    assert default_scanner.t_rise.units == ureg('ms').units

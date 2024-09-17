import numpy as np

from microtool.scanner_parameters import ScannerParameters
from microtool.utils.solve_echo_time import minimal_echo_time
from microtool.utils.unit_registry import unit


def test_minimal_echo_time():
    scanner_parameters = ScannerParameters(
        4.e-3 * unit('s'),
        6.e-3 * unit('s'),
        14.e-3 * unit('s'),
        400e-3 * unit('mT/mm'),
        1300 * unit('mT/mm/s'))
    b = np.array([0, 100, 200, 400, 800, 1600, 3200], dtype=float) * unit('s/mm²')  # s/mm^2
    expected_te = np.array([0.03508504, 0.0359497, 0.03651081, 0.0372936, 0.03838058, 0.03988306,
                            0.04194756]) * unit('s')
    te_min = minimal_echo_time(b, scanner_parameters)

    # TODO: lower tolerance when minimal_echo_time has been reviewed.
    assert np.allclose(te_min, expected_te, rtol=1e-2)

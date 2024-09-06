import numpy as np
import pytest

from microtool.scanner_parameters import ScannerParameters
from microtool.utils.solve_echo_time import minimal_echo_time


def test_minimal_echo_time():
    scanner_parameters = ScannerParameters(4.e-3, 6.e-3, 14.e-3, 400e-3, 1300)
    b = np.array([0, 100, 200, 400, 800, 1600, 3200], dtype=float)  # s/mm^2
    expected_te = [0.03508504, 0.0359497, 0.03651081, 0.0372936, 0.03838058, 0.03988306, 0.04194756]

    assert minimal_echo_time(b, scanner_parameters) == pytest.approx(expected_te, rel=1e-3)

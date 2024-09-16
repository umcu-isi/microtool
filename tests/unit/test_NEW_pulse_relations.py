import pytest
import numpy as np

from microtool.gradient_sampling import sample_uniform
from microtool.pulse_relations import compute_b_values
from microtool.scanner_parameters import ScannerParameters
from microtool.bval_delta_pulse_relations import delta_Delta_from_TE, b_val_from_delta_Delta
from microtool.utils.unit_registry import unit


# TODO: rename Delta to 'pulse_interval' and delta to 'pulse_duration'
class TestDelta_delta_PulseRelation:
    """
    Parameters and expected values obtained from MATLAB code for pulse_relations
    
    """
    
    expected_Delta = np.array([11.573, 9.773, 12.016, 18.646]) * 1e-3 * unit('s')
    expected_delta = np.array([6.881, 5.081, 7.323, 13.953]) * 1e-3 * unit('s')

    M = 4  # number of parameter combinations for every direction

    gradient_vectors = sample_uniform(M)

    echo_times = np.array([21.947, 18.347, 22.832, 36.091]) * 1e-3 * unit('s')
    scanner_parameters = ScannerParameters(
        0.6e-3 * unit('s'),
        1.2e-3 * unit('s'),
        14.e-3 * unit('s'),
        265e-3 * unit('mT/mm'),
        83 * unit('mT/mm/s')
    )

    def test_delta_echo_relation(self):
        delta, Delta = delta_Delta_from_TE(self.echo_times, self.scanner_parameters)
        
        assert delta == pytest.approx(self.expected_delta, rel=1e-4)
        assert Delta == pytest.approx(self.expected_Delta, rel=1e-4)
            

class TestB_val_PulseRelation:
    expected_b = 20087 * unit('s/mm²')

    # other pulse parameters
    Delta = 0.025 * unit('s')
    delta = 0.02 * unit('s')
    G_magnitude = 0.2 * unit('mT/mm')
    scanner_parameters = ScannerParameters(
        4e-3 * unit('s'),
        6e-3 * unit('s'),
        14e-3 * unit('s'),
        200e-3 * unit('mT/mm'),
        1300 * unit('mT/mm/s')
    )

    def test_simple_pulse_relation(self):
        """
        Testing if the parameter set from alexander 2008 can be reproduced
        """
                    
        computed_b = b_val_from_delta_Delta(self.delta, self.Delta, self.G_magnitude, self.scanner_parameters)
        assert computed_b == pytest.approx(self.expected_b, abs=1e3)
        assert computed_b.units == self.expected_b.units

        computed_b_old = compute_b_values(self.G_magnitude, self.Delta, self.delta,
                                          scanner_parameters=self.scanner_parameters)
        assert computed_b_old == pytest.approx(computed_b, rel=1e-6)
        assert computed_b_old.units == self.expected_b.units

import pytest
import numpy as np

from microtool.gradient_sampling import sample_uniform
from microtool.scanner_parameters import ScannerParameters
from microtool.bval_delta_pulse_relations import delta_Delta_from_TE, b_val_from_delta_Delta


# TODO: rename Delta to 'pulse_interval' and delta to 'pulse_duration'
class TestDelta_delta_PulseRelation:
    """
    Parameters and expected values obtained from MATLAB code for pulse_relations
    
    """
    
    expected_Delta = np.array([11.573, 9.773, 12.016, 18.646]) * 1e-3
    expected_delta = np.array([6.881, 5.081, 7.323, 13.953]) * 1e-3

    M = 4  # number of parameter combinations for every direction

    gradient_vectors = sample_uniform(M)

    echo_times = np.array([21.947, 18.347, 22.832, 36.091]) * 1e-3
    scan_parameters = ScannerParameters(0.6e-3, 1.2e-3, 14.e-3, 265e-3, 83)

    def test_delta_echo_relation(self):
        delta, Delta = delta_Delta_from_TE(self.echo_times, self.scan_parameters)
        
        assert delta == pytest.approx(self.expected_delta, rel=1e-3)
        assert Delta == pytest.approx(self.expected_Delta, rel=1e-3)
            

class TestB_val_PulseRelation:
    expected_b = 20087
    expected_b_units = 's/mm²'

    # other pulse parameters
    Delta = 0.025
    delta = 0.02
    G_magnitude = 0.2
    scan_parameters = ScannerParameters(4e-3, 6e-3, 14e-3, 200e-3, 1300)

    def test_simple_pulse_relation(self):
        """
        Testing if the parameter set from alexander 2008 can be reproduced
        """
                    
        computed_b = b_val_from_delta_Delta(self.delta, self.Delta, self.G_magnitude, self.scan_parameters)

        assert computed_b == pytest.approx(self.expected_b, abs=1e3)

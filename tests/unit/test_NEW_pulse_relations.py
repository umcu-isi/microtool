import numpy as np

from microtool.gradient_sampling import sample_uniform
from microtool.pulse_relations import compute_b_values
from microtool.scanner_parameters import ScannerParameters
from microtool.bval_delta_pulse_relations import diffusion_pulse_from_echotime, b_value_from_diffusion_pulse
from microtool.utils.unit_registry import unit


class TestDiffusionPulseFromEchotime:
    """
    Parameters and expected values obtained from MATLAB code for pulse_relations
    
    """
    
    expected_pulse_interval = np.array([11.573, 9.773, 12.016, 18.646]) * 1e-3 * unit('s')
    expected_pulse_duration = np.array([6.881, 5.081, 7.323, 13.953]) * 1e-3 * unit('s')

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

    def test_diffusion_pulse_from_echotime(self):
        pulse_duration, pulse_interval = diffusion_pulse_from_echotime(self.echo_times, self.scanner_parameters)
        
        assert np.allclose(pulse_duration, self.expected_pulse_duration, rtol=1e-4)
        assert np.allclose(pulse_interval, self.expected_pulse_interval, rtol=1e-4)
            

class TestComputeBValues:
    expected_b = 20087 * unit('s/mm²')

    # other pulse parameters
    pulse_interval = 0.025 * unit('s')
    pulse_duration = 0.02 * unit('s')
    pulse_magnitude = 0.2 * unit('mT/mm')
    scanner_parameters = ScannerParameters(
        4e-3 * unit('s'),
        6e-3 * unit('s'),
        14e-3 * unit('s'),
        200e-3 * unit('mT/mm'),
        1300 * unit('mT/mm/s')
    )

    def test_compute_b_values(self):
        """
        Testing if the parameter set from alexander 2008 can be reproduced
        """
                    
        computed_b = b_value_from_diffusion_pulse(self.pulse_duration, self.pulse_interval, self.pulse_magnitude,
                                                  self.scanner_parameters)
        assert np.allclose(computed_b, self.expected_b, atol=1e3)

        computed_b_old = compute_b_values(self.pulse_duration, self.pulse_interval, self.pulse_magnitude,
                                          scanner_parameters=self.scanner_parameters)
        assert np.allclose(computed_b_old, computed_b, rtol=1e-6)

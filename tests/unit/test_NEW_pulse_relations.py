import numpy as np

from microtool.gradient_sampling import sample_uniform
# from microtool.pulse_relations import diffusion_pulse_from_echo_time
from microtool.scanner_parameters import ScannerParameters
from microtool.bval_delta_pulse_relations import diffusion_pulse_from_echotime
from microtool.utils.unit_registry import unit


class TestDiffusionPulseFromEchotime:
    """
    Parameters and expected values obtained from MATLAB code for pulse_relations
    
    """
    
    expected_pulse_interval = np.array([11.573, 9.773, 12.016, 18.646]) * 1e-3 * unit('s')
    expected_pulse_duration = np.array([6.881, 5.081, 7.323, 13.953]) * 1e-3 * unit('s')

    M = 4  # number of parameter combinations for every direction

    gradient_vectors = sample_uniform(M)

    # TODO: a half readout time of 14 ms and t180 of 1.2 ms requires a TE of at least 29.2 ms, so the first three TE's
    #  are impossible. Where do these numbers come from?
    echo_times = np.array([21.947, 18.347, 22.832, 36.091]) * 1e-3 * unit('s')
    scanner_parameters = ScannerParameters(
        0.6e-3 * unit('s'),
        1.2e-3 * unit('s'),
        14.e-3 * unit('s'),
        265e-3 * unit('mT/mm'),
        83 * unit('mT/mm/s')
    )

    def test_diffusion_pulse_from_echotime(self):
        pulse_duration, pulse_interval = diffusion_pulse_from_echotime(
            self.echo_times, self.scanner_parameters)
        
        assert np.allclose(pulse_duration, self.expected_pulse_duration, rtol=1e-4)
        assert np.allclose(pulse_interval, self.expected_pulse_interval, rtol=1e-4)

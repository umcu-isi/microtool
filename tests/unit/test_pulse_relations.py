import numpy as np
import pytest

from microtool.scanner_parameters import ScannerParameters
from microtool.pulse_relations import diffusion_pulse_from_b_value, echo_time_from_diffusion_pulse, \
    pulse_interval_from_duration, b_value_from_diffusion_pulse, pulse_magnitude_from_b_value, \
    diffusion_pulse_from_echo_time
from microtool.utils.unit_registry import unit


scanner_parameters = ScannerParameters(
    t_90=2e-3 * unit('s'),
    t_180=4e-3 * unit('s'),
    t_half=14e-3 * unit('s'),
    g_max=200e-3 * unit('mT/mm'),  # From Alexander (2008), Table 1. [DOI: 10.1002/mrm.21646]
    s_max=1300 * unit('mT/mm/s'),  # typical is 1300 T/m/s (small bore)
)

expected_b = 21.0051e3 * unit('s/mm²')  # Alexander (2008), Table 1 reads 20087 s/mm² (lower slew rate?)

# Other pulse parameters from Alexander (2008), Table 1. [DOI: 10.1002/mrm.21646]
pulse_interval = 25e-3 * unit('s')
pulse_duration = 20e-3 * unit('s')
pulse_magnitude = 200e-3 * unit('mT/mm')


def test_b_value():
    b = b_value_from_diffusion_pulse(pulse_duration, pulse_interval, pulse_magnitude, scanner_parameters)
    assert b.units == expected_b.units
    assert b.magnitude == pytest.approx(expected_b.magnitude, rel=1e-5)


def test_get_gradients():
    g = pulse_magnitude_from_b_value(expected_b, pulse_duration, pulse_interval, scanner_parameters)
    assert g.units == pulse_magnitude.units
    assert g.magnitude == pytest.approx(pulse_magnitude.magnitude, rel=1e-5)


class TestDiffusionPulseFromEchoTime:
    expected_pulse_interval = np.array([17.0, 17.25, 26.0, 101.0]) * 1e-3 * unit('s')
    expected_pulse_duration = np.array([0.0, 0.125, 8.84615, 83.84615]) * 1e-3 * unit('s')
    expected_pulse_magnitude = np.array([0.0, 162.5, 200.0, 200.0]) * 1e-3 * unit('mT/mm')

    echo_times = np.array([32.0, 32.5, 50.0, 200.0]) * 1e-3 * unit('s')

    def test_diffusion_pulse_from_echo_time(self):
        pulse_duration, pulse_interval, pulse_magnitude = diffusion_pulse_from_echo_time(
            self.echo_times, scanner_parameters)

        assert np.allclose(pulse_duration, self.expected_pulse_duration, rtol=1e-9)
        assert np.allclose(pulse_interval, self.expected_pulse_interval, rtol=1e-9)
        assert np.allclose(pulse_magnitude, self.expected_pulse_magnitude, rtol=1e-9)


def test_echo_time_too_short():
    # 2×t_half + t_180 = 32 ms, so TE = 31 ms is impossible.
    echo_times = np.array([31.0]) * 1e-3 * unit('s')
    with pytest.raises(ValueError):
        diffusion_pulse_from_echo_time(echo_times, scanner_parameters)


def test_chained_pulse_relations():
    for g_max in np.array([0.01, 0.04, 0.4]) * unit('mT/mm'):  # Test different maximum gradient magnitudes.
        for t_half in np.array([0.001, 0.014, 0.1]) * unit('s'):  # Test different half readout times (<> t_90).
            for s_max in np.array([15, 150, 1500]) * unit('mT/mm/s'):  # Test different slew rates.
                scanner_parameters = ScannerParameters(
                    t_90=4e-3 * unit('s'),
                    t_180=6e-3 * unit('s'),
                    t_half=t_half,
                    g_max=g_max,
                    s_max=s_max,
                )
                # print(f"G_max = {g_max:.1e}, t_half = {t_half:.1e}, s_max = {s_max:.1e}")

                # Test a chain of pulse relations. The output should have minimal echo time, match input and not violate
                # any constraints.
                b_init = np.array([0, 0.01, 0.1, 1, 10, 100, 1000, 10000]) * unit('s/mm²')
                pulse_duration, pulse_interval, pulse_magnitude = diffusion_pulse_from_b_value(
                    b_init, scanner_parameters)
                # print(f"δ = {pulse_duration:.3e}, Δ = {pulse_interval:.3e}, G = {pulse_magnitude:.3e}")

                # Test if the computed b-values are as expected.
                b = b_value_from_diffusion_pulse(pulse_duration, pulse_interval, pulse_magnitude, scanner_parameters)
                assert np.allclose(b, b_init, rtol=1e-9)  # Default relative tolerance is 1e-5

                # Test if the computed magnitudes are as expected.
                g = pulse_magnitude_from_b_value(b_init, pulse_duration, pulse_interval, scanner_parameters)
                assert np.allclose(g, pulse_magnitude, rtol=1e-9)  # Default relative tolerance is 1e-5

                # Test the constraint G <= g_max
                assert np.all(pulse_magnitude <= scanner_parameters.g_max)

                # Test the constraint Δ >= δ + t_rise + t_180
                t_rise = pulse_magnitude / scanner_parameters.s_max
                min_interval = pulse_duration + t_rise + scanner_parameters.t_180
                assert np.all(pulse_interval >= min_interval)

                # Calculate the minimal echo times for these diffusion sequences.
                te = echo_time_from_diffusion_pulse(pulse_duration, pulse_interval, pulse_magnitude, scanner_parameters)

                # Calculating the optimal diffusion pulse for this echo time should give the same duration, interval and
                # magnitude:
                pulse_duration_te, pulse_interval_te, pulse_magnitude_te = diffusion_pulse_from_echo_time(
                    te, scanner_parameters)
                assert np.allclose(pulse_duration_te, pulse_duration, rtol=1e-9)  # Default relative tolerance is 1e-5
                assert np.allclose(pulse_interval_te, pulse_interval, rtol=1e-9)  # Default relative tolerance is 1e-5
                assert np.allclose(pulse_magnitude_te, pulse_magnitude, rtol=1e-9)  # Default relative tolerance is 1e-5

                # Shorter pulse durations require longer intervals to reach the same b-value or they are impossible.
                incl = b_init > 0  # Exclude b=0 in the next tests.
                h = 1e-6  # Relative difference in pulse duration.
                duration_other = (1 - h) * pulse_duration[incl]  # Slightly shorter duration
                try:
                    interval_other = pulse_interval_from_duration(duration_other, pulse_magnitude[incl], b_init[incl],
                                                                  scanner_parameters)
                    assert np.all((interval_other > pulse_interval[incl]))

                    # Shorter (suboptimal) pulse durations result in longer echo times.
                    te_other = echo_time_from_diffusion_pulse(duration_other, interval_other, pulse_magnitude[incl],
                                                              scanner_parameters)
                    assert np.all(te_other > te[incl])

                    # But the b-values should be the same.
                    b = b_value_from_diffusion_pulse(duration_other, interval_other, pulse_magnitude[incl],
                                                     scanner_parameters)
                    assert np.allclose(b, b_init[incl], rtol=1e-9)  # Default relative tolerance is 1e-5

                except ValueError:
                    pass

                # Longer pulse durations require shorter intervals to reach the same b-value.
                duration_other = (1 + h) * pulse_duration[incl]  # Slightly longer duration
                interval_other = pulse_interval_from_duration(duration_other, pulse_magnitude[incl], b_init[incl],
                                                              scanner_parameters)
                assert np.all((interval_other < pulse_interval[incl]))

                # Longer (suboptimal) pulse durations result in longer echo times.
                te_other = echo_time_from_diffusion_pulse(duration_other, interval_other, pulse_magnitude[incl],
                                                          scanner_parameters)
                assert np.all(te_other > te[incl])

                # But the b-values should be the same.
                b = b_value_from_diffusion_pulse(duration_other, interval_other, pulse_magnitude[incl],
                                                 scanner_parameters)
                assert np.allclose(b, b_init[incl], rtol=1e-9)  # Default relative tolerance is 1e-5

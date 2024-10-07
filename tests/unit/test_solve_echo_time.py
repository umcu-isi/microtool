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



# TODO: t_rise is often the time to reach maximum gradient strength, but when the pulse magnitude is known it could be
#  shorter.
#

# * diffusion_pulse_from_echotime(echo_times: np.array, scanner_parameters: ScannerParameters)
# Computes pulse durations and intervals from TE. Assumes maximum gradient strength and maximum slew rate.
#   >> Implementation questionable.
#
# * b_value_from_diffusion_pulse(pulse_duration, pulse_interval, pulse_magnitude, scanner_parameters: ScannerParameters)
# Computes b-values from pulse duration, interval and magnitude.
#   >> Same as compute_b_values?
#   >> Use pulse_magnitude instead of g_max?
#
# * get_gradients(b, pulse_duration, pulse_interval, scanner_parameters: ScannerParameters)
# Computes gradient magnitudes from b values, pulse duration, and interval.
#   >> TESTING: this should match output from b_value_from_diffusion_pulse and vice-versa
#
# * compute_b_values(pulse_duration: np.ndarray, pulse_interval: np.ndarray, pulse_magnitude: np.ndarray,
#                    scanner_parameters: Optional[ScannerParameters] = None)
# Computes b-values from pulse duration, interval and magnitude
#   >> Same as b_value_from_diffusion_pulse?
#
# * compute_t_rise(pulse_magnitude: np.ndarray, scanner_parameters: ScannerParameters)
# Computes rise-time from pulse magnitude and maximum slew rate
#
# * New_minimal_echo_time(scanner_parameters: ScannerParameters)
# Computes minimal echo time given T90, T180 and maximum rise time.
#   >> Does not take gradient duration and interval into account!
#
# * minimal_echo_time(b, scanner_parameters: ScannerParameters)
# Computes minimal echo time given T90, T180 and pulse duration.
#   >> Incorrect results (due to errors in compute_delta_max)
#
# * echo_time(pulse_duration, b, scanner_parameters: ScannerParameters)
# Computes TE from pulse duration, b-value, T90 and rise time.
#
# * compute_delta_max(b, scanner_parameters: ScannerParameters)
# Computes the maximum possible pulse duration for a given b-value. Used in minimal_echo_time.
#   >> Contains errors


# b_test
# TE = ~minimal_echo_time(b)
# δ, Δ = ~diffusion_pulse_from_echotime(TE)
# g_max = get_gradients(b, δ, Δ)  # Should return g_max!
# b = b_value_from_diffusion_pulse(δ, Δ, g_max)  # Should return b_test!
# b = compute_b_values(δ, Δ, g_max)  # Should return b_test!
# TE = echo_time(δ, b)  # Should return the same echo time!


# t_rise = compute_t_rise(g_max)  # Should return t_rise_max
#


# find δ that minimizes the TE for a given b-value

# TE = echo_time(δ, b)  # echo time for a given b-value
# b = b_value_from_diffusion_pulse(δ, Δ, g_max)
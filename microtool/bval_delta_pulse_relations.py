from typing import Tuple

import numpy as np

from .pulse_relations import b_value_from_diffusion_pulse, diffusion_pulse_from_echo_time
from .scanner_parameters import ScannerParameters
from .constants import B_VAL_UB


def constrained_dependencies(dependency: list, parameters: dict, scanner_parameters: ScannerParameters):
    """
    Checks if the scanner parameters meet the constraints. Returns a boolean array with True for each parameter that
    meets all constraints and False otherwise.
    """

    n_params = len(parameters['echo_times'])
    constraints = np.ones(n_params, dtype=bool)

    # Based on B-Value dependency uniquely
    if 'DiffusionPulseDuration' not in dependency or 'DiffusionPulseInterval' not in dependency:
        # Constrain b-values to TE. This computation is based on maximal G.
        pulse_duration, pulse_interval, pulse_magnitude = diffusion_pulse_from_echo_time(
            parameters['echo_times'], scanner_parameters)
        b_from_te = b_value_from_diffusion_pulse(pulse_duration, pulse_interval, pulse_magnitude, scanner_parameters)
        constraints &= parameters['b_values'] < b_from_te

    else:
        # Constrain deltas to TE.
        # In this case deltas are optimize and used directly to compute b_val.

        pulse_duration, pulse_interval, pulse_magnitude = diffusion_pulse_from_echo_time(
            parameters['echo_times'], scanner_parameters)
        constraints &= parameters['pulse_durations'] < pulse_duration
        constraints &= parameters['pulse_intervals'] < pulse_interval

        b_values = b_value_from_diffusion_pulse(parameters['pulse_durations'],
                                                parameters['pulse_intervals'],
                                                parameters['pulse_magnitudes'],
                                                scanner_parameters)
        constraints &= b_values < B_VAL_UB

    return constraints


def diffusion_pulse_from_echotime(echo_times: np.array,
                                  scanner_parameters: ScannerParameters) -> Tuple[np.array, np.array]:
    trise = 0
    t90 = scanner_parameters.t_90
    t180 = scanner_parameters.t_180

    # TODO: trise is zero. Does this make sense?
    if trise > 0.5 * t90:
        tau1 = trise
        tau2 = trise - 0.5 * t90
    else:
        tau1 = 0.5 * t90
        tau2 = tau1
        
    t_rise = scanner_parameters.g_max / scanner_parameters.s_max
        
    pulse_durations = echo_times / 2 - t180 / 2 - tau1 - t_rise
    pulse_intervals = pulse_durations + t_rise + t180 + tau2          
    
    return pulse_durations, pulse_intervals


# # TODO: the same as compute_b_values?
# # TODO: use pulse_magnitude instead of g_max?
# def b_value_from_diffusion_pulse(pulse_duration, pulse_interval, pulse_magnitude,
#                                  scanner_parameters: ScannerParameters) -> np.array:
#     g_max = scanner_parameters.g_max
#     s_max = scanner_parameters.s_max
#
#     t_rise = g_max / s_max
#
#     b_vals = GAMMA**2 * pulse_magnitude**2 * (
#             pulse_duration**2 * (pulse_interval - pulse_duration / 3) +
#             (t_rise**3) / 30 -
#             (pulse_duration * t_rise**2) / 6
#     )
#
#     return b_vals

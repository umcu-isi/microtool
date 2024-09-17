"""
All the times in this module are computed in ms. so we scale them at the final function.
"""
from typing import Tuple

import numpy as np

from .scanner_parameters import ScannerParameters
from .constants import GAMMA, B_VAL_UB


def constrained_dependencies(dependency: list, parameters: dict, scanner_parameters: ScannerParameters):
    """
    Checks if the scanner parameters meet the constraints. Returns a boolean array with True for each parameter that
    meets all constraints and False otherwise.
    """

    n_params = len(parameters['echo_times'])
    constraints = np.ones(n_params, dtype=bool)

    # Based on B-Value dependency uniquely
    if 'DiffusionPulseWidth' not in dependency or 'DiffusionPulseInterval' not in dependency:
        # Constrain b-values to TE. This computation is based on maximal G.
        delta, Delta = delta_Delta_from_TE(parameters['echo_times'], scanner_parameters)
        b_from_te = b_val_from_delta_Delta(delta, Delta, scanner_parameters.g_max, scanner_parameters)
        constraints &= parameters['b_values'] < b_from_te

    else:
        # Constrain deltas to TE.
        # In this case deltas are optimize and used directly to compute b_val.

        delta_from_te, Delta_from_TE = delta_Delta_from_TE(parameters['echo_times'], scanner_parameters)
        constraints &= parameters['pulse_widths'] < delta_from_te
        constraints &= parameters['pulse_intervals'] < Delta_from_TE

        b_values = b_val_from_delta_Delta(parameters['pulse_widths'],
                                          parameters['pulse_intervals'],
                                          parameters['gradient_magnitudes'],
                                          scanner_parameters)
        constraints &= b_values < B_VAL_UB

    return constraints


def delta_Delta_from_TE(echo_times: np.array, scanner_parameters: ScannerParameters) -> Tuple[np.array, np.array]:
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
        
    t_ramp = scanner_parameters.g_max / scanner_parameters.s_max
        
    pulse_widths = echo_times / 2 - t180 / 2 - tau1 - t_ramp
    pulse_intervals = pulse_widths + t_ramp + t180 + tau2          
    
    return pulse_widths, pulse_intervals


# TODO: the same as compute_b_values?
def b_val_from_delta_Delta(delta, Delta, G, scanner_parameters: ScannerParameters) -> np.array:
    g_max = scanner_parameters.g_max
    s_max = scanner_parameters.s_max
    
    t_ramp = g_max / s_max

    # TODO: Frank uses the term -(pulse_width / 6) * t_rise**2
    b_vals = GAMMA**2 * G**2 * (delta**2 * (Delta - delta/3) + (t_ramp**3) / 30 - (delta * t_ramp**2) / 6)

    return b_vals

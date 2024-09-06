"""
All the times in this module are computed in ms. so we scale them at the final function.
"""
from copy import copy
from typing import Tuple

import numpy as np

from .scanner_parameters import ScannerParameters
from .constants import GAMMA, B_VAL_UB


def constrained_dependencies(dependency: list, parameters: dict, scan_parameters: ScannerParameters):
    """
    Checks if the scanner parameters meet the constraints. Returns a boolean array with True for each parameter that
    meets all constraints and False otherwise.
    """

    n_params = len(parameters['echo_times'])
    constraints = np.ones(n_params, dtype=bool)

    # Based on B-Value dependency uniquely
    if 'DiffusionPulseWidth' not in dependency or 'DiffusionPulseInterval' not in dependency:
        # Constrain b-values to TE. This computation is based on maximal G.
        delta, Delta = delta_Delta_from_TE(parameters['echo_times'], scan_parameters)
        b_from_te = b_val_from_delta_Delta(delta, Delta, scan_parameters.G_max, scan_parameters)
        constraints &= parameters['b_values'] < b_from_te

    else:
        # Constrain deltas to TE.
        # In this case deltas are optimize and used directly to compute b_val.

        delta_from_te, Delta_from_TE = delta_Delta_from_TE(parameters['echo_times'], scan_parameters)
        constraints &= parameters['pulse_widths'] < delta_from_te
        constraints &= parameters['pulse_intervals'] < Delta_from_TE

        b_values = b_val_from_delta_Delta(parameters['pulse_widths'],
                                          parameters['pulse_intervals'],
                                          parameters['gradient_magnitudes'],
                                          scan_parameters)
        constraints &= b_values < B_VAL_UB

    return constraints


def delta_Delta_from_TE(echo_times: np.array, scanner_parameters: ScannerParameters) -> Tuple[np.array, np.array]:
    # To [ms]
    scanner_parameters = copy(scanner_parameters)
    scan_parameter_to_ms_um(scanner_parameters)
    
    trise = 0
    t90 = scanner_parameters.t_90
    t180 = scanner_parameters.t_180
    
    if trise > 0.5 * t90:
        tau1 = trise
        tau2 = trise - 0.5 * t90
    else:
        tau1 = 0.5 * t90
        tau2 = tau1
        
    t_ramp = scanner_parameters.G_max / scanner_parameters.S_max
        
    pulse_widths = (echo_times * 1e3) / 2 - t180 / 2 - tau1 - t_ramp
    pulse_intervals = pulse_widths + t_ramp + t180 + tau2          
    
    return pulse_widths * 1e-3, pulse_intervals * 1e-3  # Return back to s


def b_val_from_delta_Delta(delta, Delta, G, scan_parameters: ScannerParameters) -> np.array:
    # to '1/mT . 1/ms'
    gamma = GAMMA * 1e-3
    
    #To [ms]
    scanner_parameters = copy(scan_parameters)
    scan_parameter_to_ms_um(scanner_parameters)

    G_max = scanner_parameters.G_max  #mT/um
    S_max = scanner_parameters.S_max  #mT/um/ms
    
    delta = delta * 1e3  #to ms
    Delta = Delta * 1e3  #to ms
    
    t_ramp = G_max / S_max  #ms

    b_vals = gamma**2 * G**2 * (delta**2 * (Delta - delta/3) + (t_ramp**3) / 30 - (delta * t_ramp**2) / 6)
    
    #From ms/um^2 to s/mm^2
    return b_vals * 1e-3


# TODO: Remove this function. An object should always have values in known units, at any time.
def scan_parameter_to_ms_um(scanner_parameters: ScannerParameters):
    # time parameters
    scanner_parameters.t_90 *= 1e3
    scanner_parameters.t_half *= 1e3
    scanner_parameters.t_180 *= 1e3
    # inverse time parameter
    
    #From mT/mm/s to mT/um/ms
    scanner_parameters.S_max *= 1e-6
    
    #Cristina 03-05
    #Convert from mT/mm to mT/um
    scanner_parameters.G_max *= 1e-3 

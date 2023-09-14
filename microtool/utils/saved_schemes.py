import numpy as np

from microtool.acquisition_scheme import InversionRecoveryAcquisitionScheme, DiffusionAcquisitionScheme
from microtool.constants import PULSE_TIMING_LB, PULSE_TIMING_UB
from microtool.gradient_sampling import sample_uniform_half_sphere
from microtool.scanner_parameters import default_scanner


def make_repeated_measurements(val_list, N_rep):
    return np.concatenate([np.repeat(val, N_rep) for val in val_list])


def alexander_optimal_perturbed() -> DiffusionAcquisitionScheme:
    N = 30  # Number of measurement directions
    M = 4  # "measurements" i.e. unique acquisition parameter combinations
    N_pulses = N * M

    G_max = 0.2  # T m^-1
    default_scanner.G_max = G_max

    gradient_magnitudes = make_repeated_measurements([0.2, 0.2, 0.121, 0.2], N)
    gradient_directions = np.tile(sample_uniform_half_sphere(N), (M, 1))
    pulse_intervals = make_repeated_measurements([0.025, 0.026, 0.029, 0.013], N)
    pulse_widths = make_repeated_measurements([0.02, 0.018, 0.016, 0.008], N)
    scheme = DiffusionAcquisitionScheme(gradient_magnitudes, gradient_directions, pulse_widths, pulse_intervals,
                                        scan_parameters=default_scanner)

    # fix echo time to max values
    scheme["EchoTime"].values = np.repeat(PULSE_TIMING_UB, N_pulses)
    scheme["EchoTime"].set_fixed_mask(np.ones(N_pulses, dtype=bool))

    # mark the repeated parameters
    repeated_parameters = ['DiffusionPulseMagnitude', 'DiffusionPulseWidth', 'DiffusionPulseInterval']
    for parameter in repeated_parameters:
        scheme[parameter].set_repetition_period(N)

    return scheme


def alexander_initial_random() -> DiffusionAcquisitionScheme:
    N = 30  # Number of measurement directions
    M = 4  # "measurements" i.e. unique acquisition parameter combinations
    N_pulses = N * M
    G_max = 0.2  # T m^-1
    default_scanner.G_max = G_max
    gradient_magnitudes = np.random.uniform(0.0, G_max, N_pulses)
    gradient_directions = np.tile(sample_uniform_half_sphere(N), (M, 1))

    pulse_intervals = np.linspace(PULSE_TIMING_LB, 0.02, N_pulses) + 0.01
    pulse_widths = np.linspace(PULSE_TIMING_LB, 0.01, N_pulses)

    scheme = DiffusionAcquisitionScheme(gradient_magnitudes, gradient_directions, pulse_widths, pulse_intervals,
                                        scan_parameters=default_scanner)

    # fix echo time to max values
    scheme["EchoTime"].values = np.repeat(PULSE_TIMING_UB, N_pulses)
    scheme["EchoTime"].set_fixed_mask(np.ones(N_pulses, dtype=bool))

    # mark the repeated parameters
    repeated_parameters = ['DiffusionPulseMagnitude', 'DiffusionPulseWidth', 'DiffusionPulseInterval']
    for parameter in repeated_parameters:
        scheme[parameter].set_repetition_period(N)

    return scheme


def ir_scheme_repeated_parameters(n_pulses: int) -> InversionRecoveryAcquisitionScheme:
    """
    helper function for making scheme with different number of pulses.

    :param n_pulses:
    :return: A not so decent IR acquisition scheme
    """
    tr = np.repeat(500, n_pulses)
    te = np.repeat(20, n_pulses)
    ti = np.repeat(400, n_pulses)
    return InversionRecoveryAcquisitionScheme(tr, te, ti)


def ir_scheme_increasing_parameters(n_pulses: int) -> InversionRecoveryAcquisitionScheme:
    """
    helper function for making scheme with different number of pulses.

    :param n_pulses:
    :return: A decent IR acquisition scheme
    """
    tr = np.repeat(500, n_pulses)
    te = np.linspace(10, 20, n_pulses)
    ti = np.linspace(50, 400, n_pulses)
    return InversionRecoveryAcquisitionScheme(tr, te, ti)

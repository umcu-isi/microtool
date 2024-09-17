import numpy as np

from .unit_registry import unit
from ..acquisition_scheme import InversionRecoveryAcquisitionScheme, DiffusionAcquisitionScheme
from ..constants import PULSE_TIMING_LB, PULSE_TIMING_UB
from ..gradient_sampling import sample_uniform_half_sphere
from ..scanner_parameters import default_scanner


def alexander_b0_measurement(eps_time: float = 1e-3, eps_gradient: float = 1e-2):
    N = 30  # Number of measurement directions
    M = 4  # "measurements" i.e. unique acquisition parameter combinations
    N_pulses = (N * M + 1)
    default_scanner = get_scanner_parameters()
    gradient_directions, gradient_magnitudes, pulse_intervals, pulse_widths = get_scheme_parameters_perturbed(M, N,
                                                                                                              eps_gradient,
                                                                                                              eps_time)

    gradient_directions, gradient_magnitudes, pulse_intervals, pulse_widths = insert_b0_measurement(gradient_directions,
                                                                                                    gradient_magnitudes,
                                                                                                    pulse_intervals,
                                                                                                    pulse_widths)

    scheme = DiffusionAcquisitionScheme(gradient_magnitudes * unit('mT/mm'),
                                        gradient_directions,
                                        pulse_widths * unit('s'),
                                        pulse_intervals * unit('s'),
                                        scanner_parameters=default_scanner)

    scheme.fix_b0_measurements()
    fix_echo_times(N_pulses, scheme)
    handle_repeated_parameters(N, scheme)
    return scheme


def alexander_optimal_perturbed(eps_time: float = 1e-3, eps_gradient: float = 1e-2) -> DiffusionAcquisitionScheme:
    N = 30  # Number of measurement directions
    M = 4  # "measurements" i.e. unique acquisition parameter combinations
    # Total number of measurements including the added b0 measurement
    N_pulses = (N * M)

    default_scanner = get_scanner_parameters()
    gradient_directions, gradient_magnitudes, pulse_intervals, pulse_widths = get_scheme_parameters_perturbed(M, N,
                                                                                                              eps_gradient,
                                                                                                              eps_time)

    scheme = DiffusionAcquisitionScheme(gradient_magnitudes * unit('mT/mm'),
                                        gradient_directions,
                                        pulse_widths * unit('s'),
                                        pulse_intervals * unit('s'),
                                        scanner_parameters=default_scanner)
    fix_echo_times(N_pulses, scheme)

    # mark the repeated parameters
    handle_repeated_parameters(N, scheme)

    return scheme


def alexander_initial_random() -> DiffusionAcquisitionScheme:
    N = 30  # Number of measurement directions
    M = 4  # "measurements" i.e. unique acquisition parameter combinations
    N_pulses = N * M
    default_scanner = get_scanner_parameters()
    gradient_directions, gradient_magnitudes, pulse_intervals, pulse_widths = get_scheme_parameters_random(M, N,
                                                                                                           N_pulses)

    scheme = DiffusionAcquisitionScheme(gradient_magnitudes * unit('s'),
                                        gradient_directions,
                                        pulse_widths * unit('s'),
                                        pulse_intervals * unit('s'),
                                        scanner_parameters=default_scanner)

    # fix echo time to max values
    fix_echo_times(N_pulses, scheme)

    # mark the repeated parameters
    handle_repeated_parameters(N, scheme)
    return scheme


def ir_scheme_repeated_parameters(n_pulses: int) -> InversionRecoveryAcquisitionScheme:
    """
    helper function for making scheme with different number of pulses.

    :param n_pulses:
    :return: A not so decent IR acquisition scheme
    """
    tr = np.repeat(500, n_pulses) * unit('s')
    te = np.repeat(20, n_pulses) * unit('s')
    ti = np.repeat(400, n_pulses) * unit('s')
    return InversionRecoveryAcquisitionScheme(tr, te, ti)


def ir_scheme_increasing_parameters(n_pulses: int) -> InversionRecoveryAcquisitionScheme:
    """
    helper function for making scheme with different number of pulses.

    :param n_pulses:
    :return: A decent IR acquisition scheme
    """
    tr = np.repeat(500, n_pulses) * unit('s')
    te = np.linspace(10, 20, n_pulses) * unit('s')
    ti = np.linspace(50, 400, n_pulses) * unit('s')
    return InversionRecoveryAcquisitionScheme(tr, te, ti)


def insert_b0_measurement(gradient_directions, gradient_magnitudes, pulse_intervals, pulse_widths):
    gradient_magnitudes = np.insert(gradient_magnitudes, 0, 0.0)
    pulse_widths = duplicate_first_measurement(pulse_widths)
    pulse_intervals = duplicate_first_measurement(pulse_intervals)
    gradient_directions = duplicate_first_measurement(gradient_directions)
    return gradient_directions, gradient_magnitudes, pulse_intervals, pulse_widths


def get_scanner_parameters():
    g_max = 0.2 * unit('mT/mm')  # T m^-1
    default_scanner.g_max = g_max
    # Alexander assumes zero rise time of infinite slewrate
    default_scanner.s_max = np.inf * unit('mT/mm/s')
    default_scanner.t_180 = 0.005 * unit('s')
    return default_scanner


def get_scheme_parameters_perturbed(M, N, eps_gradient, eps_time):
    gradient_magnitudes = make_repeated_measurements([0.2, 0.2, 0.121, 0.2], N) - eps_gradient
    # prepending a b0 measurement
    gradient_directions = np.tile(sample_uniform_half_sphere(N), (M, 1))
    pulse_intervals = make_repeated_measurements([0.025, 0.026, 0.029, 0.013], N) + eps_time
    pulse_widths = make_repeated_measurements([0.020, 0.018, 0.016, 0.0079999], N) - eps_time
    return gradient_directions, gradient_magnitudes, pulse_intervals, pulse_widths


def get_scheme_parameters_random(M, N, N_pulses):
    g_max = 0.2
    gradient_magnitudes = np.random.uniform(0.0, g_max, N_pulses)
    gradient_directions = np.tile(sample_uniform_half_sphere(N), (M, 1))
    pulse_intervals = np.linspace(PULSE_TIMING_LB, 0.02, N_pulses) + 0.01
    pulse_widths = np.linspace(PULSE_TIMING_LB, 0.01, N_pulses)
    return gradient_directions, gradient_magnitudes, pulse_intervals, pulse_widths


def fix_echo_times(N_pulses, scheme):
    # fix echo time to max values
    scheme["EchoTime"].values = np.repeat(PULSE_TIMING_UB, N_pulses) * unit('s')
    scheme["EchoTime"].set_fixed_mask(np.array([True]*N_pulses))


def handle_repeated_parameters(N, scheme):
    repeated_parameters = ['DiffusionPulseMagnitude', 'DiffusionPulseWidth', 'DiffusionPulseInterval']
    for parameter in repeated_parameters:
        scheme[parameter].set_repetition_period(N)


def make_repeated_measurements(val_list, N_rep):
    return np.concatenate([np.repeat(val, N_rep) for val in val_list])


def duplicate_first_measurement(array):
    return np.insert(array, 0, array[0], axis=0)

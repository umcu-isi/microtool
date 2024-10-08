from typing import Tuple, Union

import numpy as np

from .constants import GAMMA
from .scanner_parameters import ScannerParameters
from .utils.math import real_cbrt, newton_polynomial_root


EPSILON = 1 + 1e-12  # Δ and δ are multiplied by this value in sanity checks.


def pulse_magnitude_from_b_value(b: np.ndarray, pulse_duration: np.ndarray, pulse_interval: np.ndarray,
                                 scanner_parameters: ScannerParameters) -> np.ndarray:
    """
    Compute diffusion pulse magnitudes from b-values, pulse intervals (∆), durations (δ).

    :param b: b-value in [s/mm²]
    :param pulse_duration: diffusion pulse duration [s]
    :param pulse_interval: diffusion pulse interval [s]
    :param scanner_parameters: scanner parameter definition
    :return: numpy array with gradient magnitudes [mT/mm]
    """
    # For trapezoidal gradient pulses, b = γ²G²(δ²(Δ − δ/3) + ξ³/30 − δ ξ²/6), where ξ is the rise time G / s_max

    # Solving for G gives a fifth order polynomial:
    #   [Eq 1]   0 = (γ² / s_max³ / 30) G⁵ − (γ² δ / s_max² / 6) G⁴ + γ²δ²(Δ − δ/3) G² - b
    # Since there is most likely no analytical solution, we use Newton's method to find the root. Because the term for
    # G⁵ is significantly smaller, we can use the solution for
    #   [Eq 2]   0 = −(γ² δ / s_max² / 6) G'⁴ + γ²δ²(Δ − δ/3) G'² - b
    # as an initial guess. Solving Eq 2 for G'² gives:
    #   [Eq 3] G'² = (√(c2² + 4 c4 b) - c2) / (2 c4) , where
    #   [Eq 4]  c2 = γ²δ²(Δ − δ/3)  # Second-order term
    #   [Eq 5]  c4 = -γ² δ / s_max² / 6  # Fourth-order term

    s_max = scanner_parameters.s_max

    c5 = GAMMA ** 2 / (30 * s_max ** 3)
    c4 = - GAMMA ** 2 * pulse_duration / (6 * s_max ** 2)  # [Eq 5]
    c2 = (GAMMA * pulse_duration) ** 2 * (pulse_interval - pulse_duration / 3)  # [Eq 4]
    c0 = -b
    x0 = np.nan_to_num(np.sqrt((np.sqrt(c2 ** 2 + 4 * c4 * b) - c2) / (2 * c4)))  # Square root of [Eq 3]

    pulse_magnitude = newton_polynomial_root([c0, None, c2, None, c4, c5], x0, n=3)  # 3 iterations are sufficient

    t_rise = pulse_magnitude / s_max

    if np.any(EPSILON * pulse_duration < t_rise):
        raise ValueError("Pulse duration shorter than rise time.")
    if np.any(EPSILON * pulse_interval < pulse_duration + t_rise + scanner_parameters.t_180):
        raise ValueError("b-value too low, pulse interval too short, or pulse duration too long.")

    return pulse_magnitude


def b_value_from_diffusion_pulse(pulse_duration: Union[np.ndarray, float],
                                 pulse_interval: Union[np.ndarray, float],
                                 pulse_magnitude: Union[np.ndarray, float],
                                 scanner_parameters: ScannerParameters) -> Union[np.ndarray, float]:
    """
    Compute b-values from diffusion pulse intervals (∆), durations (δ) and magnitudes.

    :param pulse_duration: Pulse durations [s]
    :param pulse_interval: Pulse intervals [s]
    :param pulse_magnitude: Pulse magnitudes [mT/mm]
    :param scanner_parameters: scanner parameter definition
    :return: b-values [s/mm²]
    """
    t_rise = pulse_magnitude / scanner_parameters.s_max

    if np.any(EPSILON * pulse_duration < t_rise):
        raise ValueError("Pulse duration shorter than rise time.")
    if np.any(EPSILON * pulse_interval < pulse_duration + t_rise + scanner_parameters.t_180):
        raise ValueError("Pulse interval too short or pulse duration too long.")

    # For trapezoidal gradient pulses, b = γ²G²(δ²(Δ − δ/3) + ξ³/30 − δ ξ²/6), where ξ is the rise time.
    # See the 'Advanced Discussion' on https://mriquestions.com/what-is-the-b-value.html

    term1 = pulse_duration ** 2 * (pulse_interval - pulse_duration / 3)
    term2 = t_rise ** 3 / 30
    term3 = (pulse_duration * t_rise ** 2) / 6

    return (GAMMA * pulse_magnitude)**2 * (term1 + term2 - term3)


def diffusion_pulse_from_b_value(b: np.ndarray, scanner_parameters: ScannerParameters) ->\
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given a series of b-values, calculate the diffusion pulse duration, interval and magnitude for which the echo time
     can be as short as possible.

    :param b: An array of b-values [s/mm²]
    :param scanner_parameters: Scanner parameter definition
    :return: Gradient pulse duration [s], interval [s], and magnitude [mT/mm]
    """
    #        90°             180°         readout
    #   RF _|¯¯|___________|¯¯¯|_______________________
    #   RO ___________________________|¯¯¯¯¯¯¯¯¯¯¯¯¯|__
    # Diff _____/¯¯¯¯\__________/¯¯¯¯\_________________
    #
    #           |-------Δ-------|     |t_half|
    #           |-δ-|           |-δ-|
    #         |-| t_90/2            |-| ξ
    #
    # For trapezoidal gradient pulses, b = γ²G²(δ²(Δ − δ/3) + ξ³/30 − δ ξ²/6), where ξ is the rise time. The lowest
    # feasible echo time is reached on the intersection of
    #   [Eq 1]  Δ(δ) = c/δ² + δ/3 + 1/6 ξ²/δ  with
    #           Δ(δ) = δ + ξ + u , where
    #   [Eq 2]     c = b/(γG)² - ξ³/30
    #   [Eq 3]     u = t_180 + |t_half - t_90/2|  # 'Free space'
    #
    # Rearranging gives a third order polynomial for δ:
    #   [Eq 4]     0 = δ³ + 3/2 (ξ + t_180 + t_half - t_90/2) δ² - 1/4 ξ²δ - 3/2 c
    #
    # Furthermore, these constraints apply:
    #    (1)      δ >= ξ  (diffusion pulse duration cannot be shorter than the rise time)
    #    (2)  TE(δ) >= t_180 + 2 δ + 2 ξ + 2 t_half  (2nd diffusion pulse should fit inbetween 180° and read-out)
    #
    # To meet constraint (1) G might have to be decreased.
    # Constraint (2) just requires increasing the echo time, which does not affect the diffusion pulses.

    # TODO: is there any realistic scenario in which solving -2c/δ³ - 1/6 ξ²/δ² + 4/3 = 0 (the derivative of TE(δ)
    #  to δ) DOES NOT violate the constraint Δ(δ) > δ + t_180 + ξ ?

    t_90 = scanner_parameters.t_90
    t_180 = scanner_parameters.t_180
    g_max = scanner_parameters.g_max
    s_max = scanner_parameters.s_max
    t_half = scanner_parameters.t_half
    t_rise = g_max / s_max

    c = (b / (GAMMA * g_max) ** 2 - t_rise ** 3 / 30)  # [Eq 2]
    u = t_180 + np.abs(t_half - 0.5 * t_90)  # [Eq 3] 'Free space'

    # Calculate the pulse durations by solving Eq 4.
    pulse_duration = real_cbrt(1.5 * (t_rise + u), -0.25 * t_rise ** 2, -1.5 * c)  # [Eq 4]
    pulse_interval = c / pulse_duration ** 2 + pulse_duration / 3 + t_rise ** 2 / (6 * pulse_duration)  # [Eq 1]
    pulse_magnitude = g_max * np.ones_like(pulse_duration)

    # Identify solutions that violate the constraint δ >= ξ
    invalid = ~(pulse_duration >= t_rise)  # δ might be NaN, hence the ~(δ >= ξ) instead of (δ < ξ).

    if np.any(invalid):
        # Use δ = ξ, ξ = G / s_max (triangular pulse with maximum slew rate) and Δ(δ) = 2ξ + u to calculate the pulse
        # magnitude G resulting in b.
        # Solving b = γ²G²(δ²(Δ − δ/3) + ξ³/30 − δ ξ²/6) for ξ with those definitions gives a fifth order polynomial:
        #       2 ξ + u = (b / (γ s_max ξ)² - ξ³ / 30) / ξ² + ξ/3 + ξ² / (6 ξ)
        #             0 = b / (γ² s_max² ξ⁴) - 23 ξ / 15 - u
        #   [Eq 5]    0 = b / (γ s_max)² - 23 ξ⁵ / 15 - u ξ⁴
        # Since there is most likely no analytical solution, we use Newton's method to find the root. Because the term
        # for ξ⁵ is significantly smaller, we can use the solution for
        #   [Eq 6]    0 = b / (γ s_max)² - u ξ'⁴
        # as an initial guess. Solving Eq 6 for ξ' gives:
        #   [Eq 7]   ξ' = ∜(b / (γ s_max)²)

        c5 = -23 / 15  # Fifth-order term in Eq 5.
        c4 = -u  # Fourth-order term in Eq 5.
        c0 = b[invalid] / (GAMMA * s_max)**2  # Zeroth-order term in Eq 5.
        x0 = (c0 / u) ** 0.25  # [Eq 7]
        t_rise = newton_polynomial_root([c0, None, None, None, c4, c5], x0, n=5)

        pulse_duration[invalid] = t_rise
        pulse_magnitude[invalid] = t_rise * s_max

        # Calculate the pulse interval. Prevent division by t_rise = 0, in which case Δ = u.
        zeros = t_rise == 0
        t_rise[zeros] = u  # Dummy value, anything other than 0 is fine.
        c = (b[invalid] / (GAMMA * t_rise * s_max) ** 2 - t_rise ** 3 / 30)  # [Eq 2]
        interval = c / t_rise ** 2 + t_rise / 3 + t_rise ** 2 / (6 * t_rise)  # [Eq 1]
        interval[zeros] = u  # Δ = u where t_rise = 0
        pulse_interval[invalid] = interval

    return pulse_duration, pulse_interval, pulse_magnitude


def pulse_interval_from_duration(pulse_duration: np.ndarray, pulse_magnitude: np.ndarray, b: np.ndarray,
                                 scanner_parameters: ScannerParameters) -> np.ndarray:
    # For trapezoidal gradient pulses, b = γ²G²(δ²(Δ − δ/3) + ξ³/30 − δ ξ²/6), where ξ is the rise time. Solving for Δ
    # gives:
    #   [Eq 1]  Δ(δ) = c/δ² + δ/3 + 1/6 ξ²/δ , where
    #   [Eq 2]     c = b/(γG)² - ξ³/30
    t_rise = pulse_magnitude / scanner_parameters.s_max

    if np.any(EPSILON * pulse_duration < t_rise):
        raise ValueError("Pulse duration shorter than rise time.")

    c = (b / (GAMMA * pulse_magnitude) ** 2 - t_rise ** 3 / 30)  # [Eq 2]
    pulse_interval = c / pulse_duration ** 2 + pulse_duration / 3 + t_rise ** 2 / (6 * pulse_duration)  # [Eq 1]

    if np.any(EPSILON * pulse_interval < pulse_duration + t_rise + scanner_parameters.t_180):
        raise ValueError("Pulse duration too long or b-value too low.")

    return pulse_interval


def echo_time_from_diffusion_pulse(pulse_duration: np.ndarray, pulse_interval: np.ndarray, pulse_magnitude: np.ndarray,
                                   scanner_parameters: ScannerParameters) -> np.ndarray:
    """
    Calculates the minimum echo time for a diffusion pulse sequence with a specific pulse duration, interval, and
     magnitude.

    :param pulse_duration: Pulse durations [s]
    :param pulse_interval: Pulse intervals [s]
    :param pulse_magnitude: Pulse magnitudes [mT/mm]
    :param scanner_parameters: Scanner parameter definition
    :return: Minimum echo times [s]
    """
    t_90 = scanner_parameters.t_90
    t_180 = scanner_parameters.t_180
    t_half = scanner_parameters.t_half
    t_rise = pulse_magnitude / scanner_parameters.s_max

    if np.any(EPSILON * pulse_duration < t_rise):
        raise ValueError("Pulse duration shorter than rise time.")
    if np.any(EPSILON * pulse_interval < pulse_duration + t_rise + t_180):
        raise ValueError("Pulse interval too short or pulse duration too long.")

    # The minimum echo time is either limited by the pulse interval (long interval) or by the pulse duration (short
    # interval).
    #
    # Long interval:
    #        90°             180°         readout
    #   RF _|¯¯|___________|¯¯¯|_______________________
    #   RO ___________________________|¯¯¯¯¯¯¯¯¯¯¯¯¯|__
    # Diff _____/¯¯\______________/¯¯\_________________
    #
    # Short interval:
    #        90°             180°         readout
    #   RF _|¯¯|___________|¯¯¯|_______________________
    #   RO ___________________________|¯¯¯¯¯¯¯¯¯¯¯¯¯|__
    # Diff _________/¯¯¯¯\______/¯¯¯¯\_________________
    #
    # Short interval, short readout:
    #        90°       180°      readout
    #   RF _|¯¯|______|¯¯¯|____________________________
    #   RO ________________________|¯|_________________
    # Diff _____/¯¯¯¯\______/¯¯¯¯\_____________________
    #
    te_interval = 0.5 * t_90 + pulse_interval + pulse_duration + t_rise + t_half
    te_duration = 2 * (0.5 * t_180 + pulse_duration + t_rise + np.maximum(t_half, 0.5 * t_90))

    return np.maximum(te_interval, te_duration)


def diffusion_pulse_from_echo_time(echo_time: np.ndarray, scanner_parameters: ScannerParameters) ->\
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given a series of echo-times, calculate the diffusion pulse duration, interval and magnitude for which the b-value
     is as high as possible.

    :param echo_time: An array of echo times [s]
    :param scanner_parameters: Scanner parameter definition
    :return: Gradient pulse duration [s], interval [s], and magnitude [mT/mm]
    """
    t_90 = scanner_parameters.t_90
    t_180 = scanner_parameters.t_180
    g_max = scanner_parameters.g_max
    s_max = scanner_parameters.s_max
    t_half = scanner_parameters.t_half
    t_rise = g_max / s_max

    # A pulse-interval-limited sequence gives the highest b-values. In this case the echo time is:
    #   [Eq 1]   TE = t_90 / 2 + Δ + δ + ξ + t_half
    # The optimal b-value is reached when
    #   [Eq 2]    Δ = δ + ξ + u , where
    #   [Eq 3]    u = t_180 + |t_half - t_90/2|  # 'Free space'
    # Substituting Eq 2 in Eq1 and solving for δ gives:
    #   [Eq 4]    δ = (TE - u - t_half - t_90 / 2)/2 - ξ

    u = t_180 + np.abs(t_half - 0.5 * t_90)  # [Eq 3] 'Free space'
    pulse_duration = 0.5 * (echo_time - u - t_half - 0.5 * t_90) - t_rise  # [Eq 4]
    pulse_interval = pulse_duration + t_rise + u  # [Eq 2]
    pulse_magnitude = g_max * np.ones_like(pulse_duration)

    # Identify durations that violate the constraint δ >= ξ
    invalid = ~(pulse_duration >= t_rise)  # δ might be NaN, hence the ~(δ >= ξ) instead of (δ < ξ).

    if np.any(invalid):
        # Substituting δ = ξ in Eq 4 gives δ = (TE - u - t_half - t_90 / 2) / 4
        # Due to floating point arithmetic, only TE - (u + t_half + t_90 / 2) equals 0 when TE = u + t_half + t_90 / 2.
        pulse_duration[invalid] = 0.25 * (echo_time[invalid] - (u + t_half + 0.5 * t_90))
        pulse_interval[invalid] = 2 * pulse_duration[invalid] + u
        pulse_magnitude[invalid] = pulse_duration[invalid] * s_max

    if np.any(pulse_duration < 0):
        # The echo time is less than max(2×t_half, t_90) + t_180, which is impossible.
        raise ValueError("Echo time too short.")

    return pulse_duration, pulse_interval, pulse_magnitude

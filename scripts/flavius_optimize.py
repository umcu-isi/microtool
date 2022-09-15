"""
Here we optimize the scheme according to flavius' specifications:
"""
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from microtool.acquisition_scheme import FlaviusAcquisitionScheme
from microtool.optimize import optimize_scheme
from microtool.tissue_model import FlaviusSignalModel
from microtool.utils.solve_echo_time import minimal_echo_time

# dont print so many digits
np.set_printoptions(formatter={'float': '{: .2e}'.format})

# -------- Defining the signal model
# white matter diffusivity [mm^2/s]
D = 0.8 * 1e-3
# White matter T2 relaxation time [ms]
T2 = 80
signal_model = FlaviusSignalModel(T2, D)
# -------- Defining Acquisition scheme

scan_parameters = {
    'excitation_time_half_pi': 4,  # ms
    'excitation_time_pi': 6,  # ms
    'half_readout_time': 14,  # ms
    'max_gradient': 200e-3,  # mT/mm
    'max_slew_rate': 1300e-3  # mT/mm/ms
}

# b_values [s/mm^2]
b = np.array(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 10,
     10, 10, 10, 10, 10, 10]) * 1e3

# Echo times [ms]
TE = np.array(
    [30, 55, 57, 59, 68, 74, 90, 110, 125, 150, 30, 55, 57, 59, 68, 74, 90, 110, 125, 150, 55, 57, 59, 68, 74, 90, 110,
     125, 150, 57, 59, 68, 74, 90, 110, 125, 150, 59, 68, 74, 90, 110, 125, 150])

scheme = FlaviusAcquisitionScheme(b, TE, **scan_parameters)
scheme_copy = deepcopy(scheme)

# ----------- Optimization
optimizer_options = {"maxiter": 1000, 'disp': True}

scheme_optimized, _ = optimize_scheme(scheme, signal_model,
                                      noise_variance=0.02, repeat=1, method='SLSQP', options=optimizer_options)

print('Optimized Scheme: \n', scheme_optimized)

# -------------- Plotting
plt.figure('Signal')
plt.plot(signal_model(scheme_copy), '.', label="Initial")
plt.plot(signal_model(scheme_optimized), '.', label='Optimized')
plt.xlabel('M')
plt.ylabel('S/S0')
plt.legend()
plt.tight_layout()

plt.figure('b_values')
plt.plot(scheme_copy.b_values, '.', label="Initial")
plt.plot(scheme_optimized.b_values, '.', label="Optimized")

plt.figure('b vs TE')
plt.plot(scheme_copy.echo_times, scheme_copy.b_values, '.', label='Initial')
plt.plot(scheme_optimized.echo_times, scheme_optimized.b_values, '.', label='Optimised')

t180 = scheme_optimized['PulseDurationPi'].values
t90 = scheme_optimized['PulseDurationHalfPi'].values
G_max = scheme_optimized['MaxPulseGradient'].values
t_rise = scheme_optimized['RiseTime'].values
t_half = scheme_optimized['HalfReadTime'].values

# Changing to ms/mm^2 for computing minimal echo time.
b_plot = np.linspace(0.1, 15e3, num=500) * 1e3
plt.plot(minimal_echo_time(b_plot, t90, t180, t_half, G_max, t_rise), b_plot * 1e-3, label="Analytic lowerbound")
plt.xlabel("TE")
plt.ylabel('b')
plt.legend()

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from dmipy.signal_models.gaussian_models import G2Zeppelin

from microtool.dmipy import DmipyMultiTissueModel
from microtool.acquisition_scheme import DiffusionAcquisitionSchemeBValue
from microtool.optimize.optimize import iterative_shell_optimization
from microtool.pulse_relations import diffusion_pulse_from_echo_time, b_value_from_diffusion_pulse
from microtool.scanner_parameters import default_scanner
from microtool.tissue_model import RelaxationTissueModel
from microtool.utils.unit_registry import unit

model_dmipy = G2Zeppelin(mu=[0.5, 0.5], lambda_par=0.5e-09, lambda_perp=0.2e-09)

model = DmipyMultiTissueModel(model_dmipy)
model._dmipy_fix_parameters('G2Zeppelin_1_mu', [0.5, 0.5])
model._dmipy_fix_parameters('G2Zeppelin_1_lambda_par', 0.5e-09)
model._dmipy_fix_parameters('G2Zeppelin_1_lambda_perp', 0.2e-09)

# TODO: The comment says T2 in s, but the docstring reads T2: Transverse relaxation time constant T2 in milliseconds.
#  Which one is correct?
model_relaxed = RelaxationTissueModel(model, t2=0.020 * unit('s'))  # T2 in s, 20 ms in MATLAB

n_shells = 4
n_directions = 15

#Provide a first random initialization
model_dependencies = model_relaxed.get_dependencies()
initial_scheme = DiffusionAcquisitionSchemeBValue.random_shell_initialization(
    n_shells, n_directions, model_dependencies)

iterations = 10
solver_options = {"verbose": 2, "maxiter": 10}

optimal_scheme = iterative_shell_optimization(
    initial_scheme,
    model_relaxed,
    n_shells,
    n_directions,
    iterations=iterations,
    noise_variance=0.02,
    method="trust-constr",
    solver_options=solver_options
)


# PLOT RESULTS --------------------------------------------------------------------

b_vals_optimal = optimal_scheme.b_values 
echo_times_optimal = optimal_scheme.echo_times

scanner_parameters = default_scanner
minTE = max(2 * scanner_parameters.t_half, scanner_parameters.t_90) + scanner_parameters.t_180
maxTE = 0.05 * unit('s')  # Maximum TE
step = 0.001 * unit('s')  # Step size
TEs = np.arange(minTE, maxTE, step)
pulse_duration, pulse_interval, pulse_magnitude = diffusion_pulse_from_echo_time(TEs, scanner_parameters)
bs = b_value_from_diffusion_pulse(pulse_duration, pulse_interval, pulse_magnitude, scanner_parameters)

plt.figure()
# Scatter plot of xall_(:,2) vs xall_(:,1)
plt.scatter(echo_times_optimal, b_vals_optimal, color='red')  # Optimal scheme
# Plot b-values
plt.plot(TEs, bs)

# Show plot
max_iter = solver_options["maxiter"]
plt.title(f"{iterations} iterations, maxiter {max_iter}")
plt.xlabel('TE [s]')
plt.ylabel('b [s/mm^2]')
plt.show()

import numpy as np

from dmipy.signal_models.gaussian_models import G2Zeppelin
from microtool.dmipy import DmipyTissueModel

from microtool.acquisition_scheme import DiffusionAcquisitionScheme_bval_dependency
from microtool.optimize.optimize import iterative_shell_optimization
from microtool.scanner_parameters import default_scanner
from microtool.tissue_model import RelaxationTissueModel


model_dmipy = G2Zeppelin(mu=[0.5, 0.5], lambda_par=0.5e-09, lambda_perp=0.2e-09)

model = DmipyTissueModel(model_dmipy)
model._dmipy_fix_parameters('G2Zeppelin_1_mu', [0.5, 0.5])
model._dmipy_fix_parameters('G2Zeppelin_1_lambda_par', 0.5e-09)
model._dmipy_fix_parameters('G2Zeppelin_1_lambda_perp', 0.2e-09)

# TODO: The comment says T2 in s, but the docstring reads T2: Transverse relaxation time constant T2 in milliseconds.
#  Which one is correct?
model_relaxed = RelaxationTissueModel(model, T2=0.020) #T2 in s, 20 ms in MATLAB

n_shells = 4
n_directions = 15

#Provide a first random initialization
model_dependencies = model_relaxed.get_dependencies()
initial_scheme = DiffusionAcquisitionScheme_bval_dependency.random_shell_initialization(
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


#PLOT RESULTS --------------------------------------------------------------------

import matplotlib.pyplot as plt

b_vals_optimal = optimal_scheme.b_values 
echo_times_optimal = optimal_scheme.echo_times

from microtool.utils.solve_echo_time import New_minimal_echo_time
from microtool.bval_delta_pulse_relations import delta_Delta_from_TE, b_val_from_delta_Delta

scan_parameters = default_scanner
minTE = New_minimal_echo_time(scan_parameters)
maxTE = 0.05  # Maximum TE
step = 0.001  # Step size
TEs = np.arange(minTE, maxTE, step)
delta, Delta = delta_Delta_from_TE(TEs, scan_parameters)
bs = b_val_from_delta_Delta(delta, Delta, scan_parameters.G_max, scan_parameters)

plt.figure()
# Scatter plot of xall_(:,2) vs xall_(:,1)
plt.scatter(echo_times_optimal, b_vals_optimal, color='red') #Optimal scheme
# Plot b-values
plt.plot(TEs, bs)

# Show plot
max_iter = solver_options["maxiter"]
plt.title(f"{iterations} iterations, maxiter {max_iter}")
plt.xlabel('TE [s]')
plt.ylabel('b [s/mm^2]')
plt.show()

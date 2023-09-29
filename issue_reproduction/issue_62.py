import numpy as np
from dmipy.signal_models.gaussian_models import G1Ball

from microtool.dmipy import make_microtool_tissue_model

# setting up a simple model in dmipy, where lambda iso represents the diffusivity
simple_ball_dmipy = G1Ball(lambda_iso=1.7e-9)

# converting it to a microtool compatible model
simple_ball_microtool = make_microtool_tissue_model(simple_ball_dmipy)

from microtool.acquisition_scheme import DiffusionAcquisitionScheme
from microtool.gradient_sampling import sample_uniform_half_sphere

M = 10
N = 30

b_vals = np.concatenate([np.repeat(0, M), np.repeat(1000, M), np.repeat(3000, M)])
pulse_widths = np.concatenate([np.repeat(0.019, M), np.repeat(0.016, M), np.repeat(0.007, M)])
pulse_intervals = np.concatenate([np.repeat(0.030, M), np.repeat(0.027, M), np.repeat(0.020, M)])

directions = sample_uniform_half_sphere(N)
initial_scheme = DiffusionAcquisitionScheme.from_bvals(b_values=b_vals, b_vectors=directions, pulse_widths=pulse_widths,
                                                       pulse_intervals=pulse_intervals)

initial_scheme.fix_b0_measurements()
initial_scheme["EchoTime"].fixed = True
print(initial_scheme)

from microtool.optimize import optimize_scheme

optimal_scheme, _ = optimize_scheme(initial_scheme, simple_ball_microtool, noise_variance=0.02, method="trust-constr",
                                    solver_options={"verbose": 2})

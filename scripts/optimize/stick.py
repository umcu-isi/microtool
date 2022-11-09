"""
Optimizing a simple acquisition scheme
"""
import numpy as np

from microtool.acquisition_scheme import DiffusionAcquisitionScheme
from microtool.gradient_sampling import sample_uniform
from microtool.optimize import optimize_scheme
from microtool.utils import IO
from microtool.utils import saved_models

if __name__ == "__main__":
    # ---- loading the basic stick model
    stick = saved_models.stick()

    # ---- setting up the initial acquisition scheme
    M = 10  # number of measurements
    Mb0 = int(M / 10)
    Mb = M - Mb0
    b_values = np.concatenate([np.repeat(0, Mb0), np.linspace(1000, 5000, num=Mb)])
    b_vectors = sample_uniform(M)
    pulse_widths = np.repeat(20, M)
    pulse_intervals = np.repeat(40, M)

    scheme = DiffusionAcquisitionScheme(b_values, b_vectors, pulse_widths, pulse_intervals)
    IO.save_pickle(scheme, "schemes/stick_scheme_start.pkl")
    # ---- optimizing the scheme
    noise_var = 0.02

    scheme_opt, _ = optimize_scheme(scheme, stick, noise_variance=noise_var)

    IO.save_pickle(scheme_opt, "schemes/stick_scheme_optimal.pkl")
    print(scheme)
    print(scheme_opt)

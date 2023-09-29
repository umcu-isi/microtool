from pathlib import Path

import numpy as np

from microtool.monte_carlo.IO import make_expirement_directories
from microtool.optimize import optimize_scheme
from microtool.utils.IO import save_pickle
from microtool.utils.saved_models import cylinder_zeppelin
from microtool.utils.saved_schemes import alexander_optimal_perturbed,alexander_b0_measurement

# Making experiment file structure
experiment_name = "exp2_alexander_local_optimization"
_, modeldir, _, schemedir = make_expirement_directories('.', experiment_name)

# making initial scheme
initial_scheme = alexander_b0_measurement()
save_pickle(initial_scheme, schemedir / "initial_scheme.pkl")

# making the model
model = cylinder_zeppelin(orientation=[np.pi / 2, 0.0])
save_pickle(model, modeldir / "model.pkl")

optimal_scheme, raw_optimize_result = optimize_scheme(initial_scheme, model, noise_variance=.02, method="trust-constr",
                                                      solver_options={"verbose": 2, "maxiter": 100})

# saving result
save_pickle(optimal_scheme, schemedir / "optimal_scheme.pkl")
save_pickle(raw_optimize_result, Path(experiment_name) / "raw_optimization_result.pkl")

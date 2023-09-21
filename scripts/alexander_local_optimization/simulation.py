import numpy as np
from scipy import stats

from microtool.monte_carlo import MonteCarloSimulation
from microtool.monte_carlo.IO import get_experiment_subdirs
from microtool.utils.IO import get_pickle

plotdir, modeldir, simdir, schemedir = get_experiment_subdirs("../exp1_alexander_local_optimization")

# load model
model = get_pickle(modeldir / "model.pkl")

# load schemes
optimal_scheme = get_pickle(schemedir / "optimal_scheme.pkl")
initial_scheme = get_pickle(schemedir / "initial_scheme.pkl")

# noise setup
noise_variance = 0.002
noise_distribution = stats.norm(loc=0, scale=np.sqrt(noise_variance))
simulation = MonteCarloSimulation(optimal_scheme, model, noise_distribution, n_sim=10)
simulation.run()

from matplotlib import pyplot as plt

from microtool.monte_carlo.parameter_distributions import plot_parameter_distributions
from microtool.utils.IO import get_pickle
from microtool.utils.plotting import plot_acquisition_parameters

# loading monte carlo results
non_optimal = get_pickle("results/VERDICT/data/verdictnon-optimal_nsim=1000_norm_sigma=0.04.pkl")
gt = get_pickle("results/VERDICT/data/verdict_gt.pkl")
non_optimal = non_optimal.astype('float64')
optimal = get_pickle("results/VERDICT/data/verdictoptimal_nsim=1000_norm_sigma=0.04.pkl")
optimal = optimal.astype('float64')

# loading the acquisition schemes
scheme_init = get_pickle("../optimize/schemes/verdict_start.pkl")
scheme_opt = get_pickle("../optimize/schemes/verdict_optimal.pkl")

# for the plot formatting
symbol_map = {'G1Ball_1_lambda_iso': r"$d_{EES}$ [m$^2$/s]",
              'C1Stick_1_mu_0': r"$\theta_{stick}$ [rad]",
              'C1Stick_1_mu_1': r"$\phi_{stick}$ [rad]",
              'partial_volume_0': r"$f_{IC}$ ",
              'partial_volume_1': r"$f_{EES}$ ",
              'partial_volume_2': r"$f_{VASC}$ "}

# plotting the distributions
plot_parameter_distributions(non_optimal, gt, symbols=symbol_map, hist_label="Non-optimal", draw_gt=False)
plot_parameter_distributions(optimal, gt, symbols=symbol_map, hist_label="Optimal")

plot_acquisition_parameters(scheme_init, label="initial")
plot_acquisition_parameters(scheme_opt, label="optimized")
plt.show()

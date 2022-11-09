from matplotlib import pyplot as plt

from microtool.monte_carlo.parameter_distributions import plot_parameter_distributions
from microtool.utils.IO import get_pickle

non_optimal = get_pickle("results/VERDICT/data/verdictnon-optimal_nsim=1000_norm_sigma=0.14.pkl")
gt = get_pickle("results/VERDICT/data/verdict_gt.pkl")
non_optimal = non_optimal.astype('float64')
optimal = get_pickle("results/VERDICT/data/verdictoptimal_nsim=1000_norm_sigma=0.14.pkl")
optimal = optimal.astype('float64')

# plotting the distributions
plot_parameter_distributions(non_optimal, gt, fig_label="Non optimal scheme")
plot_parameter_distributions(optimal, gt, fig_label="optimal scheme")
plt.show()

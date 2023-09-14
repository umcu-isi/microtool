from matplotlib import pyplot as plt

from microtool.optimize import optimize_scheme
from microtool.utils.IO import save_pickle
from microtool.utils.plotting import LossInspector
from microtool.utils.saved_models import cylinder_zeppelin
from microtool.utils.saved_schemes import alexander_optimal_perturbed


def main():
    initial_scheme = alexander_optimal_perturbed()
    model = cylinder_zeppelin()
    print(model)
    print(initial_scheme)
    loss_inspector = LossInspector(initial_scheme, model, noise_var=.02)
    loss_inspector.plot([{"DiffusionPulseWidth": 1}])
    plt.savefig("loss_landscape.png")

    optimal_scheme, _ = optimize_scheme(initial_scheme, model, noise_variance=.02, method="trust-constr",
                                        solver_options={"disp": True})

    print(optimal_scheme)
    save_pickle(optimal_scheme, "alexander_optimized.pkl")


if __name__ == "__main__":
    main()

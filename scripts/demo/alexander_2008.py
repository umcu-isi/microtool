from microtool.optimize import optimize_scheme
from microtool.utils.IO import save_pickle
from microtool.utils.saved_models import cylinder_zeppelin
from microtool.utils.saved_schemes import alexander_initial_random


def main():
    initial_scheme = alexander_initial_random()
    model = cylinder_zeppelin()
    print(initial_scheme)

    optimal_scheme, _ = optimize_scheme(initial_scheme, model, noise_variance=.02,
                                        solver_options={"strategy": "best1exp"})

    print(optimal_scheme)
    save_pickle(optimal_scheme, "alexander_optimized.pkl")


if __name__ == "__main__":
    main()

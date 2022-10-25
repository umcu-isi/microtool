"""
Here we run the dmipy monte carlo simulation for the tissuemodel described in Alexander 2008:
https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.21646

"""
import pathlib
import pickle

from scipy import stats

from microtool import monte_carlo
from microtool.utils import saved_schemes, saved_models

currentdir = pathlib.Path(__file__).parent
outputdir = currentdir / "results" / "multi_compartment"
outputdir.mkdir(exist_ok=True)


def main():
    # ------------- Setting up dmipy objects -----------
    noise_var = 0.02
    # -------------ACQUISITION-------------------
    scheme = saved_schemes.alexander2008()

    # ------MODEL-------------
    mc_model = saved_models.cylinder_zeppelin()
    print("Using the following model:\n", mc_model)

    # ----------- Optimizing the scheme ------------------
    mc_model.optimize(scheme, noise_var)
    print("Using the optimized scheme:\n", scheme)
    scheme.print_acquisition_info
    # ------------ Monte Carlo --------------------
    # Setting up the noise distribution

    noise_distribution = stats.norm(loc=0, scale=noise_var)

    # Running monte carlo simulation
    n_sim = 2

    tissue_parameters = monte_carlo.run(scheme, mc_model, noise_distribution, n_sim, cascade=True)

    with open(outputdir / "alexander_shells_n_sim_{}_noise_{}.pkl".format(n_sim, noise_var), "wb") as f:
        pickle.dump(tissue_parameters, f)

    with open(outputdir / "alexander2008_ground_truth.pkl", 'wb') as f:
        pickle.dump(mc_model, f)


if __name__ == "__main__":
    main()

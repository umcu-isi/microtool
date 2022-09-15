import pathlib
import pickle

from scipy import stats
from tqdm import tqdm

from microtool import monte_carlo
from microtool.utils import saved_schemes, saved_models

currentdir = pathlib.Path(__file__).parent
outputdir = currentdir / "results"
outputdir.mkdir(exist_ok=True)


def main():
    # ------------- Setting up dmipy objects -----------
    noise_var = 0.02

    # ------MODEL-------------
    mc_model = saved_models.cylinder_zeppelin()

    # -------------ACQUISITION-------------------
    # Length 3 Integer partitions of 120 under investigation
    partitions = [[40, 40, 40], [30, 30, 60], [20, 20, 80], [8, 32, 80]]
    for shells in tqdm(partitions, desc='Partition progressbar'):
        scheme = saved_schemes.alexander2008_optimized_directions(shells)

        # ------------ Monte Carlo --------------------
        # Setting up the noise distribution
        noise_distribution = stats.norm(loc=0, scale=noise_var)

        # Running monte carlo simulation
        n_sim = 100

        tissue_parameters = monte_carlo.run(scheme, mc_model, noise_distribution, n_sim, cascade=True)

        with open(outputdir / "alexander_shells_{}_n_sim_{}_noise_{}.pkl".format(shells, n_sim, noise_var), "wb") as f:
            pickle.dump(tissue_parameters, f)

        with open(outputdir / "alexander2008_ground_truth.pkl", 'wb') as f:
            pickle.dump(mc_model, f)


if __name__ == "__main__":
    main()

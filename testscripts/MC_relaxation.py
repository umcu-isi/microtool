import pandas as pd
from matplotlib import pyplot as plt

import microtool.monte_carlo
import numpy as np
from scipy import stats
import pathlib

currentdir = pathlib.Path('.')
outputdir = currentdir / "MC_results"
outputdir.mkdir(exist_ok=True)


def main():
    # Loading the tissuemodel
    relaxation_model = microtool.tissue_model.RelaxationTissueModel(t1=900, t2=90)

    tr = np.array([500, 500, 500, 500, 500, 500, 500, 500])
    te = np.array([10, 10, 10, 10, 20, 20, 20, 20])
    ti = np.array([50, 100, 150, 200, 250, 300, 350, 400])

    # Setting and optimizing the inversion recovery scheme
    noise_var = 0.02
    ir_scheme = microtool.acquisition_scheme.InversionRecoveryAcquisitionScheme(tr, te, ti)
    relaxation_model.optimize(ir_scheme, noise_var)

    # setting noise distribution for monte carlo simulation
    noise_distribution = stats.norm(loc=0, scale=noise_var)

    # Running monte carlo simulation
    n_sim = 10000

    posterior = microtool.monte_carlo.run(ir_scheme, relaxation_model, noise_distribution, n_sim=n_sim)
    posterior = pd.DataFrame(posterior)
    # TODO: consider binary filetype for dataframe storage
    posterior.to_csv(outputdir / "TPD_relaxation.csv")

    posterior.diff().hist()
    plt.figure()
    posterior["T2"].hist()
    plt.title(f"n_sim = {n_sim}")
    plt.savefig(outputdir / "T2_distribution.png")
    plt.show()


if __name__ == "__main__":
    main()

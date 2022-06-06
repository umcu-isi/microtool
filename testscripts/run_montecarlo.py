
import microtool.monte_carlo
import numpy as np
from scipy import stats

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
noise_distribution = stats.norm(loc = 0, scale = noise_var)

# Running monte carlo simulation
posterior, covariance_matrices = microtool.monte_carlo.run(ir_scheme, relaxation_model, noise_distribution, n_sim=5)
for tissuemodel in posterior:
    print(tissuemodel)


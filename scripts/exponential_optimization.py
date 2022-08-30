"""
Optimizing simple exponential model.
"""
import numpy as np
from matplotlib import pyplot as plt

from microtool.optimization_methods import BruteForce, SOMA
from microtool.tissue_model import ExponentialTissueModel
from microtool.acquisition_scheme import EchoScheme
from microtool.optimize import optimize_scheme, compute_loss

# set the noise
noise = 0.8
# Aquisition scheme
TE = np.linspace(5, 40, num=50)
scheme = EchoScheme(TE)

# Tissuemodel
model = ExponentialTissueModel(T2=10)

# optimization
scheme_opt, _ = optimize_scheme(scheme, model, noise)

print(compute_loss(scheme, model, noise))

print(compute_loss(scheme_opt, model, noise))

print(scheme)
print(scheme_opt)

plt.figure('Echo times')
plt.plot(scheme.echo_times, '.', label='Pre')
plt.plot(scheme_opt.echo_times, '.', label='Post')
plt.legend()
plt.xlabel('measurement')
plt.ylabel('TE [ms]')

plt.figure('Signal')
plt.plot(model(scheme), '.', label='Pre optimization')
plt.plot(model(scheme_opt), '.', label='Post optimization')
plt.legend()
plt.xlabel('measurement')
plt.ylabel('signal')
plt.tight_layout()
plt.show()

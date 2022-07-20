"""
Here we optimize the scheme according to flavius' specifications:
"""
#TODO: Add the maximum gradient strength constraint
import matplotlib.pyplot as plt
import numpy as np

from microtool.optimize import optimize_scheme
from microtool.tissue_model_flavius import FlaviusSignalModel
from microtool.acquisition_scheme_flavius import FlaviusAcquisitionScheme

# -------- Defining the signal model
# white matter diffusivity (um^2/ms)
D = 0.8
# White matter T2 relaxation time (ms)
T2 = 80
signal_model = FlaviusSignalModel(T2, D)
# -------- Defining Acquisition scheme
# maximum gradient strength (mT/m)
gradient_strength = 200
# b_values s/micro m^2
b = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 10,
     10, 10, 10, 10, 10, 10])
# Echo times
TE = np.array([30, 55, 57, 59, 68, 74, 90, 110, 125, 150, 30, 55, 57, 59, 68, 74, 90, 110, 125, 150, 55, 57, 59, 68, 74, 90, 110,
      125, 150, 57, 59, 68, 74, 90, 110, 125, 150, 59, 68, 74, 90, 110, 125, 150])

scheme = FlaviusAcquisitionScheme(b, TE)

print("Pre optimized Scheme:\n", scheme)
plt.figure("Pre optimized signal ")
plt.plot(signal_model(scheme))

scheme_optimized, _ = optimize_scheme(scheme, signal_model, noise_variance=0.1, repeat=10)

plt.figure('Post optimized signal')
plt.plot(signal_model(scheme_optimized))

print('Optimized Scheme: \n', scheme_optimized)

plt.show()
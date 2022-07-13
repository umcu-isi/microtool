import matplotlib.pyplot as plt
import numpy as np

from microtool import optimize, acquisition_scheme, tissue_model
from microtool.utils.plotting import LossInspector
# # Inversion recovery

# ## 1. Create a tissue model specifying a T1 and T2
relaxation_model = tissue_model.RelaxationTissueModel(t1=900, t2=90)
print(relaxation_model)

# ## 2. Create an initial inversion-recovery acquisition scheme
# Initial TR = 500 ms, initial TE = 10 ms, initial TI = {50, ..., 400} ms

tr = np.array([500, 500, 500, 500])
te = np.array([10, 10, 10, 10])
ti = np.array([50, 100, 150, 200])

ir_scheme = acquisition_scheme.InversionRecoveryAcquisitionScheme(tr, te, ti)
print(ir_scheme)

plt.figure(figsize=(6, 4))
plt.plot(relaxation_model(ir_scheme), '.')
plt.xlabel('Measurement')
plt.ylabel('Signal attenuation')

# ## 3. Optimize the acquisition scheme
noise_variance = 0.1
brute = optimize.BruteForce(5)
relaxation_model.optimize(ir_scheme, noise_variance, bounds=[(0, 3000), (0, 100)])

print(ir_scheme)
plt.figure(figsize=(6, 4))
plt.plot(relaxation_model(ir_scheme), '.')
plt.xlabel('Measurement')
plt.ylabel('Signal attenuation')

lossinspector = LossInspector(optimize.crlb_loss, ir_scheme, relaxation_model, noise_var=noise_variance)
lossinspector.plot({"InversionTime": 1,"RepetitionTimeExcitation": 2})

plt.show()

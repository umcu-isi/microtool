# # dmpyi diffusion model
import numpy as np
from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.signal_models.cylinder_models import C1Stick
from matplotlib import pyplot as plt

import microtool.dmipy

# ## 1. Create a 'stick' diffusion model

dmipy_model = MultiCompartmentModel(models=[
    C1Stick(
        mu=[1, 1],  # Orientation in angles.
        lambda_par=0.001 * 1e-6  # Parallel diffusivity in m²/s.
    )
])

# ## 2. Wrap the dmipy model in a DmipyTissueModel

diffusion_model = microtool.dmipy.DmipyTissueModel(dmipy_model)
print(diffusion_model)

# ## 3. Create an initial diffusion acquisition scheme

b_values = np.array([0, 1000, 2000, 3000])  # s/mm²
b_vectors = np.array([[0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
pulse_widths = np.full(b_values.shape, 10)  # ms
pulse_intervals = np.full(b_values.shape, 30)  # ms

diffusion_scheme = microtool.acquisition_scheme.DiffusionAcquisitionScheme(b_values, b_vectors, pulse_widths,
                                                                           pulse_intervals)
print(diffusion_scheme)

plt.figure(figsize=(6, 4))
plt.plot(diffusion_model(diffusion_scheme), '.')
plt.xlabel('Measurement')
plt.ylabel('Signal attenuation')

# ## 4. Calculate the Cramer-Rao lower bound loss

jacobian = diffusion_model.jacobian(
    diffusion_scheme)  # Jacobian of the signal with respect to the relevant tissue parameters.
scales = [p.scale for p in diffusion_model.values()]  # Tissue parameter scales.
include = [p.optimize for p in diffusion_model.values()]  # Include tissue parameter in optimization?
noise_variance = 0.1
microtool.optimize.crlb_loss(jacobian, scales, include, noise_variance)

# ## 5. Optimize the acquisition scheme

optimizeresult = diffusion_model.optimize(diffusion_scheme, noise_variance)

print(diffusion_scheme)
plt.figure(figsize=(6, 4))
plt.plot(diffusion_model(diffusion_scheme), '.')
plt.xlabel('Measurement')
plt.ylabel('Signal attenuation')

# ## 6. Calculate the Cramer-Rao lower bound loss again
# It should be lower after optimizing the acquisition.

jacobian = diffusion_model.jacobian(
    diffusion_scheme)  # Jacobian of the signal with respect to the relevant tissue parameters.
scales = [p.scale for p in diffusion_model.values()]  # Tissue parameter scales.
include = [p.optimize for p in diffusion_model.values()]  # Include tissue parameter in optimization?
noise_variance = 0.1
microtool.optimize.crlb_loss(jacobian, scales, include, noise_variance)
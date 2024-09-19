# # MISST diffusion model

# ## 1. Set the path to the MISST MATLAB package
import numpy as np
from matplotlib import pyplot as plt

import microtool.misst
from microtool import optimize
from microtool.tissue_model import TissueParameter
from microtool.acquisition_scheme import DiffusionAcquisitionScheme
microtool.misst.set_misst_path(r'C:\development\MISST')

# ## 2. Create a 'Cylinder' diffusion model and wrap it in a MisstTissueModel

# In[ ]:


misst_model = {
    'di': TissueParameter(value=2e-9, scale=1e-9),  # Intrinsic diffusivity in m²/s.
    'rad': TissueParameter(value=5.2e-6, scale=1e-6, optimize=False),  # Cylinder radius in m.
    'theta': TissueParameter(value=0.1, scale=1),  # Angle from z axis
    'phi': TissueParameter(value=0.2, scale=1),  # Azimuthal angle
}

# In[ ]:


diffusion_model = microtool.misst.MisstTissueModel('Cylinder', misst_model)
print(diffusion_model)

# ## 3. Create an initial diffusion acquisition scheme

# In[ ]:


b_values = np.array([0, 500, 1000, 1500, 2000, 2500, 3000])  # [s/mm²]
b_vectors = np.array([[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]])
pulse_widths = np.full(b_values.shape, 10e-3)  # [s]
pulse_intervals = np.full(b_values.shape, 30e-3)  # [s]

diffusion_scheme = DiffusionAcquisitionScheme(b_values, b_vectors, pulse_widths, pulse_intervals)
print(diffusion_scheme)

# ## 4. Optimize the acquisition scheme

# In[ ]:


noise_variance = 0.1
optimize.optimize_scheme(diffusion_scheme, diffusion_model, noise_variance)


# In[ ]:


print(diffusion_scheme)
plt.figure(figsize=(6, 4))
plt.plot(diffusion_model(diffusion_scheme), '.')
plt.xlabel('Measurement')
plt.ylabel('Signal attenuation')

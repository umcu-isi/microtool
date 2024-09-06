
import numpy as np
from dmipy.signal_models.gaussian_models import G1Ball
from dmipy.signal_models import sphere_models, cylinder_models, gaussian_models


from microtool.dmipy import DmipyMultiTissueModel
from microtool.tissue_model import MultiTissueModel, RelaxationTissueModel

sphere_dmipy = sphere_models.S4SphereGaussianPhaseApproximation(diffusion_constant=2.0e-9, diameter=10e-6)
sphere = DmipyMultiTissueModel(sphere_dmipy)
sphere._dmipy_fix_parameters('S4SphereGaussianPhaseApproximation_1_diameter', 2.0e-9)
sphere_relax = RelaxationTissueModel(sphere, T2 = 0.020)


stick_dmipy = cylinder_models.C1Stick(mu=[np.pi / 2, np.pi / 2], lambda_par=8.0e-9)
stick = DmipyMultiTissueModel(stick_dmipy)
stick._dmipy_fix_parameters('C1Stick_1_lambda_par', 8.0e-9)
stick._dmipy_fix_parameters('C1Stick_1_mu', [np.pi / 2, np.pi / 2])
stick_relax = RelaxationTissueModel(stick, T2 = 0.020)

model = MultiTissueModel([sphere_relax, stick_relax], volume_fractions= [0.3, 0.7])

from microtool.acquisition_scheme import DiffusionAcquisitionScheme, InversionRecoveryAcquisitionScheme
from microtool.gradient_sampling import sample_uniform_half_sphere

M = 10
N = 30

b_vals = np.concatenate([np.repeat(0, M), np.repeat(1000, M), np.repeat(3000, M)])
pulse_widths = np.concatenate([np.repeat(0.019, M), np.repeat(0.016, M), np.repeat(0.007, M)])
pulse_intervals = np.concatenate([np.repeat(0.030, M), np.repeat(0.027, M), np.repeat(0.020, M)])

directions = sample_uniform_half_sphere(N)
initial_scheme = DiffusionAcquisitionScheme.from_bvals(b_values=b_vals, b_vectors=directions, pulse_widths=pulse_widths,
                                                        pulse_intervals=pulse_intervals)


initial_scheme.fix_b0_measurements()
initial_scheme["EchoTime"].fixed = True


#OPTIMIZATION ----------------------------------------------------------------------------------------------------------

from microtool.optimize import optimize_scheme

optimal_scheme, _ = optimize_scheme(initial_scheme, model, noise_variance=0.02, method="trust-constr", 
                                    solver_options={"verbose":2, "maxiter": 10})


# Generating basic signal on the new scheme
signal = model(optimal_scheme)


# #FITTING --------------------------------------------------------------------------------------------------------------
fitted_model = model.fit(optimal_scheme, signal, use_parallel_processing=False)
fitted_model.fitted_parameters

import numpy as np
from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.signal_models import cylinder_models
from matplotlib import pyplot as plt

from microtool.acquisition_scheme import ShellScheme
from microtool.dmipy import DmipyTissueModel
from microtool.optimize import optimize_scheme
from microtool.utils.gradient_sampling.shell_rigid_rotation import sample_shells_rotation


def main():
    # --------- initial acquisition scheme
    shells = [2, 20, 20, 20]
    b_values = np.repeat(np.array([0, 1000, 2000, 3000]), shells)
    b_vectors = np.concatenate(sample_shells_rotation(shells))
    pulse_widths = np.repeat(20, len(b_values))
    pulse_intervals = np.repeat(40, len(b_values))
    scheme = ShellScheme(b_values, b_vectors, pulse_widths, pulse_intervals)
    print(scheme)
    scheme.plot_shells_projected()

    # -------- Tissue model
    mu = (np.pi / 2., np.pi / 2.)  # in radians
    lambda_par = 1.7e-9  # in m^2/s
    stick = cylinder_models.C1Stick(mu=mu, lambda_par=lambda_par)
    stick_model = MultiCompartmentModel(models=[stick])
    stick_model_wrapped = DmipyTissueModel(stick_model)

    # --------- Optimization
    new_scheme, _ = optimize_scheme(scheme, stick_model_wrapped, noise_variance=0.02)
    print(new_scheme)
    # --------- plotting result
    new_scheme.plot_shells_projected()
    plt.show()


if __name__ == "__main__":
    main()

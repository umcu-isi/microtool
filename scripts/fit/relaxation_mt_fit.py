import numpy as np
from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.data import saved_acquisition_schemes
from dmipy.signal_models import cylinder_models, gaussian_models
from matplotlib import pyplot as plt

from microtool.dmipy import DmipyTissueModel, convert_dmipy_scheme2diffusion_scheme
from microtool.tissue_model import RelaxedMultiTissueModel

if __name__ == "__main__":
    acq_scheme = saved_acquisition_schemes.wu_minn_hcp_acquisition_scheme()
    acq_wrapped = convert_dmipy_scheme2diffusion_scheme(acq_scheme)
    print(acq_wrapped)
    # Cylinder orientation angles theta, phi := mu

    # noinspection DuplicatedCode
    mu = np.array([np.pi / 2, np.pi / 2])
    # Parralel diffusivity lambda_par in E-9 m^2/s (in the paper d_par)
    lambda_par = 1.7e-9
    lambda_perp = 0.2e-9

    zeppelin = gaussian_models.G2Zeppelin(mu, lambda_par, lambda_perp)
    stick = cylinder_models.C1Stick(mu, lambda_par)

    stick_wrapped = DmipyTissueModel(MultiCompartmentModel(models=[stick]))
    zeppelin_wrapped = DmipyTissueModel(MultiCompartmentModel(models=[zeppelin]))

    multi_model = RelaxedMultiTissueModel([stick_wrapped, zeppelin_wrapped], [.5, .5], relaxation_times=[10.0, 10.0])

    signal = multi_model(acq_wrapped)

    result = multi_model.fit(acq_wrapped, signal, method="DE")
    print(multi_model)
    multi_model.set_fit_parameters(result.fitted_parameters)
    print(multi_model)
    print(result.fitted_parameters)
    result.print_fit_information()

    predicted_signal = multi_model(acq_wrapped)
    plt.figure("Residuals")
    plt.plot(signal - predicted_signal, '.')
    plt.show()

import numpy as np
from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.data import saved_acquisition_schemes
from dmipy.signal_models import gaussian_models, cylinder_models

from microtool.dmipy import DmipyTissueModel, convert_dmipy_scheme2diffusion_scheme
from microtool.tissue_model import MultiTissueModel


class TestUnits:
    mu = np.array([np.pi / 2, np.pi / 2])
    # Parralel diffusivity lambda_par in E-9 m^2/s (in the paper d_par)
    lambda_par = 1.7e-9
    lambda_perp = 0.2e-9

    zeppelin = gaussian_models.G2Zeppelin(mu, lambda_par, lambda_perp)
    stick = cylinder_models.C1Stick(mu, lambda_par)
    stick_wrapped = DmipyTissueModel(MultiCompartmentModel(models=[stick]))
    zeppelin_wrapped = DmipyTissueModel(MultiCompartmentModel(models=[zeppelin]))
    multi_model = MultiTissueModel([stick_wrapped, zeppelin_wrapped], [.5, .5])

    def test_set_parameter_vector(self):
        expected = np.ones(len(self.multi_model))
        self.multi_model.set_parameters_from_vector(expected)
        actual = self.multi_model.parameter_vector
        np.testing.assert_equal(expected, actual)


class TestIntegration:
    acq_scheme = saved_acquisition_schemes.wu_minn_hcp_acquisition_scheme()
    acq_wrapped = convert_dmipy_scheme2diffusion_scheme(acq_scheme)
    # Cylinder orientation angles theta, phi := mu
    mu = np.array([np.pi / 2, np.pi / 2])
    # Parralel diffusivity lambda_par in E-9 m^2/s (in the paper d_par)
    lambda_par = 1.7e-9
    lambda_perp = 0.2e-9

    zeppelin = gaussian_models.G2Zeppelin(mu, lambda_par, lambda_perp)
    stick = cylinder_models.C1Stick(mu, lambda_par)
    stick_zeppelin = MultiCompartmentModel(models=[zeppelin, stick])
    single_model = DmipyTissueModel(stick_zeppelin, volume_fractions=[0.5, 0.5])

    stick_wrapped = DmipyTissueModel(MultiCompartmentModel(models=[stick]))
    zeppelin_wrapped = DmipyTissueModel(MultiCompartmentModel(models=[zeppelin]))
    multi_model = MultiTissueModel([stick_wrapped, zeppelin_wrapped], [.5, .5])

    def test_signal(self):
        expected_signal = self.single_model(self.acq_wrapped)
        actual_signal = self.multi_model(self.acq_wrapped)
        np.testing.assert_equal(expected_signal, actual_signal)

    def test_fit_debug(self):
        signal = self.single_model(self.acq_wrapped)
        result = self.multi_model.fit(self.acq_wrapped, signal)
        print(result.fitted_parameters)
        result.print_fit_information()

    def test_fit_integration(self):
        signal = self.single_model(self.acq_wrapped)
        expected_fit = self.single_model.fit(self.acq_wrapped, signal, use_parallel_processing=False)
        actual_fit = self.multi_model.fit(self.acq_wrapped, signal)

        assert expected_fit.fitted_parameters == actual_fit.fitted_parameters

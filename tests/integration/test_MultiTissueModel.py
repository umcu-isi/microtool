import numpy as np
import pytest
from dmipy.data import saved_acquisition_schemes
from dmipy.signal_models import gaussian_models, cylinder_models

from microtool.dmipy import DmipyMultiTissueModel, convert_dmipy_scheme2diffusion_scheme
from microtool.tissue_model import MultiTissueModel
from microtool.constants import MODEL_PREFIX, VOLUME_FRACTION_PREFIX


def test_set_parameter_vector():
    """
    testing MultiTissueModel.set_parameter_from_vector works as expected
    """
    mu = np.array([np.pi / 2, np.pi / 2])
    # Parralel diffusivity lambda_par in E-9 m^2/s (in the paper d_par)
    lambda_par = 1.7e-9
    lambda_perp = 0.2e-9

    zeppelin = gaussian_models.G2Zeppelin(mu, lambda_par, lambda_perp)
    stick = cylinder_models.C1Stick(mu, lambda_par)
    stick_wrapped = DmipyMultiTissueModel(stick)
    zeppelin_wrapped = DmipyMultiTissueModel(zeppelin)
    multi_model = MultiTissueModel([stick_wrapped, zeppelin_wrapped], [.5, .5])

    expected = np.ones(len(multi_model))
    multi_model.set_scaled_parameters(expected)
    actual = multi_model.scaled_parameter_vector
    np.testing.assert_equal(expected, actual)


class TestIntegrationWithAcquisitionScheme:
    """
    On this class we collect all tests that require an acquisition scheme in addition to a multi tissue model.
    Recommended to run acquisition scheme unit tests prior to integration.
    """
    acq_scheme = saved_acquisition_schemes.wu_minn_hcp_acquisition_scheme()
    acq_wrapped = convert_dmipy_scheme2diffusion_scheme(acq_scheme)
    # Cylinder orientation angles theta, phi := mu
    mu = np.array([np.pi / 2, np.pi / 2])
    # Parallel diffusivity lambda_par in E-9 m^2/s (in the paper d_par)
    lambda_par = 1.7e-9
    lambda_perp = 0.2e-9

    zeppelin = gaussian_models.G2Zeppelin(mu, lambda_par, lambda_perp)
    stick = cylinder_models.C1Stick(mu, lambda_par)
    single_model = DmipyMultiTissueModel([zeppelin, stick], volume_fractions=[0.5, 0.5])

    stick_wrapped = DmipyMultiTissueModel(stick)
    zeppelin_wrapped = DmipyMultiTissueModel(zeppelin)
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

        def absolute_azimuthal_angle(parameterdict):
            for key in parameterdict:
                if key.endswith('mu_1'):
                    parameterdict[key] = np.abs(parameterdict[key])

        # I have to extract the proper keys from fitted parameters to make dict comparison possible
        mt_to_dmipy_map = {
            'C1Stick_1_lambda_par': MODEL_PREFIX + '0_C1Stick_1_lambda_par',
            'C1Stick_1_mu_0': MODEL_PREFIX + '0_C1Stick_1_mu_0',
            'C1Stick_1_mu_1': MODEL_PREFIX + '0_C1Stick_1_mu_1',
            'G2Zeppelin_1_lambda_par': MODEL_PREFIX + '1_G2Zeppelin_1_lambda_par',
            'G2Zeppelin_1_lambda_perp': MODEL_PREFIX + '1_G2Zeppelin_1_lambda_perp',
            'G2Zeppelin_1_mu_0': MODEL_PREFIX + '1_G2Zeppelin_1_mu_0',
            'G2Zeppelin_1_mu_1': MODEL_PREFIX + '1_G2Zeppelin_1_mu_1',
            'partial_volume_0': VOLUME_FRACTION_PREFIX + '1',
        }

        mt_fit = actual_fit.fitted_parameters

        converted_dict = {}
        for dmipy_key, mt_key in mt_to_dmipy_map.items():
            converted_dict[dmipy_key] = mt_fit[mt_key]

        converted_dict['partial_volume_1'] = 1. - converted_dict["partial_volume_0"]

        expected_parameters = expected_fit.fitted_parameters
        absolute_azimuthal_angle(expected_parameters)
        absolute_azimuthal_angle(converted_dict)
        assert expected_parameters == pytest.approx(converted_dict, rel=0.05)

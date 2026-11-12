import numpy as np
import pytest
from dmipy_fit.data import saved_acquisition_schemes
from dmipy_fit.signal_models import gaussian_models, cylinder_models
from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_bvalues

from microtool.dmipy import DmipyMultiTissueModel, convert_dmipy_scheme2diffusion_scheme
from microtool.tissue_model import MultiTissueModel
from microtool.constants import MODEL_PREFIX, VOLUME_FRACTION_PREFIX, B_MAX


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

GAMMA = 2.6751525e8  # proton gyromagnetic ratio [rad s^-1 T^-1]


def _fibonacci_sphere(n: int) -> np.ndarray:
    """n roughly evenly spread unit vectors (deterministic)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.column_stack([np.cos(theta) * np.sin(phi),
                            np.sin(theta) * np.sin(phi),
                            np.cos(phi)])


def _in_bounds_scheme(b_shells=(500, 1000, 1500),   # s/mm²
                      n_dir=30, n_b0=3,
                      delta=0.015, Delta=0.040,      # s
                      g_max=0.07,                    # T/m  -> set to your scanner's limit
                      p180=0.005,                    # s    -> set to your scanner's P180
                      margin=0.9):
    """
    Own acquisition scheme that satisfies G <= g_max, Delta >= delta + P180 and b <= B_MAX.
    The b-values are checked against the cap that follows from the gradient limit,
    b_cap = (gamma * g_max * delta)^2 * (Delta - delta/3).
    """
    if Delta < delta + p180:
        raise ValueError(f"Delta={Delta} must be >= delta + P180 = {delta + p180}")

    b_cap = (GAMMA * g_max * delta) ** 2 * (Delta - delta / 3.0)          # s/m²
    b_max_si = float(B_MAX.to("s/m²").magnitude) if hasattr(B_MAX, "to") else float(B_MAX) * 1e6
    b_cap = margin * min(b_cap, b_max_si)

    shells = [b * 1e6 for b in b_shells]                                  # s/mm² -> s/m²
    if max(shells) > b_cap:
        raise ValueError(f"Largest shell {max(shells) / 1e6:.0f} s/mm² exceeds the cap "
                         f"{b_cap / 1e6:.0f} s/mm²; lengthen delta/Delta, raise g_max or lower the shells.")

    bvals = np.concatenate([np.zeros(n_b0)] + [np.full(n_dir, s) for s in shells])
    bvecs = np.vstack([np.tile([0.0, 0.0, 1.0], (n_b0, 1))] +
                      [_fibonacci_sphere(n_dir)] * len(shells))

    return acquisition_scheme_from_bvalues(bvals, bvecs, delta, Delta)

class TestIntegrationWithAcquisitionScheme:
    """
    On this class we collect all tests that require an acquisition scheme in addition to a multi tissue model.
    Recommended to run acquisition scheme unit tests prior to integration.
    """
    acq_scheme = _in_bounds_scheme()
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

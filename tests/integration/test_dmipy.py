"""
This file contains the tests for the microtool.dmipy submodule. In this way we assert that the wrappers work as
intended.

We want to test the following aspects:


1.) Wrapping and then converting an acquisition scheme yields the same acquisition scheme.

2.) When simulating signal using dmipytissuemodelwrapper we get the same result
as simulating signal using the same pure dmipy model and scheme.

3.) When computing the Jacobian through the finite differences method we get a result that agrees with analytical
 methods.
"""
from copy import copy
from typing import Dict

import numpy as np
from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.data import saved_acquisition_schemes
from dmipy.signal_models import cylinder_models
from dmipy.signal_models.gaussian_models import G1Ball

from microtool.acquisition_scheme import DiffusionAcquisitionScheme
from microtool.constants import BASE_SIGNAL_KEY
from microtool.dmipy import CascadeFitDmipy, get_microtool_parameters
from microtool.dmipy import convert_dmipy_scheme2diffusion_scheme, DmipyMultiTissueModel
from microtool.utils import saved_models, saved_schemes
from microtool.utils.unit_registry import unit


class AnalyticBall(DmipyMultiTissueModel):
    """
    Quick and dirty inheritance of dmipytissue model. Purpose is for testing finite differences
    """

    def __init__(self, lambda_iso: float):
        model = G1Ball(lambda_iso)
        super().__init__(model)

    def scaled_jacobian_analytic(self, scheme: DiffusionAcquisitionScheme) -> np.ndarray:
        bvals = copy(scheme.b_values) * (1000 * unit('mm/m'))**2  # convert to SI units

        s0 = self[BASE_SIGNAL_KEY].value
        d_iso = self['G1Ball_1_lambda_iso'].value * unit('m²/s')

        # the signal S = S_0 * e^{-T_E / T_2} * e^{-b * D}
        s = s0 * np.exp(-bvals * d_iso)
        s_diso = -bvals * s

        # d S / d D_iso , d S / d S_0
        jac = np.array([
            s_diso * self['G1Ball_1_lambda_iso'].scale,
            s * self[BASE_SIGNAL_KEY].scale]
        ).T
        return jac[:, self.include_optimize]


class TestModelSchemeIntegration:
    """
    All tests for which the microtool scheme and model wrappers are used together
    """
    # Acquisition aspects constant between tests
    acq_scheme = saved_acquisition_schemes.wu_minn_hcp_acquisition_scheme()
    acq_wrapped = convert_dmipy_scheme2diffusion_scheme(acq_scheme)

    # Models used in both test simulate signal and test fit
    # simplest tissuemodel available in dmipy
    mu = (np.pi / 2., np.pi / 2.)  # in radians
    lambda_par = 1.7e-9  # in m^2/s
    stick = cylinder_models.C1Stick(mu=mu, lambda_par=lambda_par)
    stick_model = MultiCompartmentModel(models=[stick])
    stick_model_wrapped = DmipyMultiTissueModel(stick)
    parameters = {'C1Stick_1_mu': mu, 'C1Stick_1_lambda_par': lambda_par}

    def test_simulate_signal(self):
        """
        Testing if simulate signal works if we wrap the dmipy objects in microtool. We use a stick model for the test.
        """
        # signal computation
        wrapped_signal = self.stick_model_wrapped(self.acq_wrapped)
        naked_signal = self.stick_model.simulate_signal(self.acq_scheme, self.parameters)

        np.testing.assert_allclose(wrapped_signal, naked_signal, atol=1e-4)

    def test_finite_differences(self):
        """
        Testing finite differences method for the computation of the jacobian
        """
        # dmipy model
        lambda_iso = 1.7e-9

        analytic_ball = AnalyticBall(lambda_iso)

        # Scheme --------------
        dmipy_scheme = saved_acquisition_schemes.wu_minn_hcp_acquisition_scheme()
        mt_scheme = convert_dmipy_scheme2diffusion_scheme(dmipy_scheme)

        jac_analytic = analytic_ball.scaled_jacobian_analytic(mt_scheme)
        jac_numeric = analytic_ball.scaled_jacobian(mt_scheme)
        # There are some rounding errors but we can reach a relative tolerance 1e-5 which should be enough.
        np.testing.assert_allclose(jac_numeric, jac_analytic, rtol=1e-5)

    def test_fit(self):
        """
        We should test if wrapped fit result is same as "naked" fit result.
        This also implicitly tests the FittedTissueModel classes. We allow a fit deviation of up to 1%.
        """
        # fit using pure dmipy objects to generate expected result
        signal = self.stick_model.simulate_signal(self.acq_scheme, self.parameters)
        pure_fit_result = self.stick_model.fit(self.acq_scheme, signal)
        expected = pure_fit_result.fitted_parameters

        # fit using wrapped objects to generate result for testing
        fit_result = self.stick_model_wrapped.fit(self.acq_wrapped, signal)
        result = fit_result.fitted_parameters

        expected_mt = get_microtool_parameters(self.stick_model_wrapped.dmipy_model, expected)

        # testing to make sure the converter works on the keys
        assert expected_mt.keys() == result.keys()

        # testing if the values are the same for all parameters
        for value, expected_value in zip(result.values(), expected_mt.values()):
            np.testing.assert_allclose(value, expected_value, rtol=0.01)

    def test_cascade_decorator(self):
        """
        Test fitting using cascaded decorator.
        """
        # -------- Acquisition
        scheme = saved_schemes.alexander_b0_measurement()

        # ---------- initialize expected result by manually sequentially fitting and setting initial values
        # generating signal
        cylinder_zeppelin = saved_models.cylinder_zeppelin([np.pi / 2, np.pi / 2])
        signal = cylinder_zeppelin(scheme)

        # fitting simple model to signal
        stick_zeppelin = saved_models.stick_zeppelin()
        simple_fit = stick_zeppelin.fit(scheme, signal, use_parallel_processing=False)
        simple_dict = simple_fit.dmipyfitresult.fitted_parameters
        # mapping fit values to initial guess or complex model
        cylinder_zeppelin.set_initial_parameters(self._stickzeppelin_to_cylinderzeppelin(simple_dict))
        expected_result = cylinder_zeppelin.fit(scheme, signal, use_parallel_processing=False)

        # ---------------- Now doing the same thing for the decorator
        # name map maps the simple model names to complex model names
        name_map = {
            # The same model so we simply map
            'G2Zeppelin_1_mu': 'G2Zeppelin_1_mu',
            'G2Zeppelin_1_lambda_par': 'G2Zeppelin_1_lambda_par',
            'G2Zeppelin_1_lambda_perp': 'G2Zeppelin_1_lambda_perp',

            # For the cylinder we initialize the orientation and parallel diffusivities
            # to those found by fitting stick zep
            "C1Stick_1_mu": 'C4CylinderGaussianPhaseApproximation_1_mu',
            "C1Stick_1_lambda_par": 'C4CylinderGaussianPhaseApproximation_1_lambda_par',

            'partial_volume_0': 'partial_volume_0',
            'partial_volume_1': 'partial_volume_1'
        }

        cylinder_zeppelin_cascade = CascadeFitDmipy(cylinder_zeppelin, stick_zeppelin, name_map)
        result = cylinder_zeppelin_cascade.fit(scheme, signal, use_parallel_processing=False)
        result_dict = result.fitted_parameters
        expected_dict = expected_result.fitted_parameters
        for parameter in result_dict.keys():
            np.testing.assert_allclose(result_dict[parameter], expected_dict[parameter])

    @staticmethod
    def _stickzeppelin_to_cylinderzeppelin(parameters: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {
            # We set the zeppelin values to the values found by stick zeppelin fitting
            'G2Zeppelin_1_mu': parameters['G2Zeppelin_1_mu'],
            'G2Zeppelin_1_lambda_par': parameters['G2Zeppelin_1_lambda_par'],
            'G2Zeppelin_1_lambda_perp': parameters['G2Zeppelin_1_lambda_perp'],

            # For the cylinder we initialize the orientation and parralel diffusivities
            # to those found by fitting stick zep
            'C4CylinderGaussianPhaseApproximation_1_mu': parameters["C1Stick_1_mu"],
            'C4CylinderGaussianPhaseApproximation_1_lambda_par': parameters["C1Stick_1_lambda_par"],

            'partial_volume_0': parameters['partial_volume_0'],
            'partial_volume_1': parameters['partial_volume_1']
        }

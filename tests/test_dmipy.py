"""
This file containts the tests for the microtool.dmipy submodule. In this way we assert that the wrappers work as
intended.

We want to test the following aspects:


1.) Wrapping and then converting an acquisition scheme yields the same acquisition scheme.

2.) When simulating signal using dmipytissuemodelwrapper we get the same result
as simulating signal using the same pure dmipy model and scheme.


"""
import numpy as np
from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.data import saved_acquisition_schemes
from dmipy.signal_models import cylinder_models

from microtool.dmipy import DmipyAcquisitionSchemeWrapper, DmipyTissueModel, convert_acquisition_scheme


def test_scheme_wrapping():
    """
    Since some of the DmipyAcquisitionScheme attributes are strange objects we test here for all float,int or
    numpy array attributes to check if they are approximately equal.

    Failure of this test means there is something wrong with microtool.dmipy.DmipyAcquisitionSchemeWrapper
    """
    # going over all attributes that are python native or numpy native and asserting equality
    # Standard acquisition scheme from dmipy.data
    naked_scheme = saved_acquisition_schemes.wu_minn_hcp_acquisition_scheme()
    # Micrtool scheme wrapper
    wrapped_scheme = DmipyAcquisitionSchemeWrapper(naked_scheme)
    # Microtool scheme converter
    converted_wrapped_scheme = convert_acquisition_scheme(wrapped_scheme)

    naked_attributes = vars(naked_scheme)
    converted_attributes = vars(converted_wrapped_scheme)
    for attribute, value in naked_attributes.items():
        report = f"The attribute {attribute} triggered an assertion, i.e., they are not equal for both " \
                 f"scheme types "
        if isinstance(value, np.ndarray):
            np.testing.assert_allclose(value, converted_attributes[attribute], rtol=0, atol=1e-5), report
        elif isinstance(value, (float, int)):
            assert value == converted_attributes[attribute], report


class TestTissueModelWrapping:
    """
    Here we collected all test concerning DmipyTissueModel class
    """
    # The pure dmipy stuff
    scheme = saved_acquisition_schemes.wu_minn_hcp_acquisition_scheme()

    mu = (np.pi / 2., np.pi / 2.)  # in radians
    lambda_par = 1.7e-9  # in m^2/s
    stick = cylinder_models.C1Stick(mu=mu, lambda_par=lambda_par)
    stick_model = MultiCompartmentModel(models=[stick])

    parameters = {'C1Stick_1_mu': mu, 'C1Stick_1_lambda_par': lambda_par}
    signal = stick_model.simulate_signal(scheme, parameters)

    # common wrapped parameters
    stick_model_wrapped = DmipyTissueModel(stick_model)

    def test_simulate_signal(self):
        """
        In this test we assert that the signal as generated by MultiCompartmentModel.simulate_signal is the same as
        the DmipyTissueModel.__call__

        In otherwords if this test is passed then DmipyTissueModel.__call__ is correct
        """

        wrapped_signal = self.stick_model_wrapped(self.scheme)

        np.testing.assert_allclose(wrapped_signal, self.signal)

    def test_fit(self):
        """
        Testing if dmipy fits properly. Fails because of orientation degeneracy ???
        """
        fitted_model = self.stick_model.fit(self.scheme, self.signal)
        for parameter, value in fitted_model.fitted_parameters.items():
            expected = np.array(self.parameters[parameter])
            np.testing.assert_allclose(np.array(value).flatten(), expected)


def test_model_scheme_integration():
    """
    This tests if the DmipyAcquisitionSchemeWrapper and the DmipyTissueModel work together to generate the correct signal
    the other tests should be passed before testing this, otherwise failure is guaranteed.
    """

    # Acquisition aspects
    acq_scheme = saved_acquisition_schemes.wu_minn_hcp_acquisition_scheme()
    acq_wrapped = DmipyAcquisitionSchemeWrapper(acq_scheme)

    # Tissuemodel aspects
    # simplest tissuemodel available in dmipy
    mu = (np.pi / 2., np.pi / 2.)  # in radians
    lambda_par = 1.7e-9  # in m^2/s
    stick = cylinder_models.C1Stick(mu=mu, lambda_par=lambda_par)
    stick_model = MultiCompartmentModel(models=[stick])
    stick_model_wrapped = DmipyTissueModel(stick_model)
    parameters = {'C1Stick_1_mu': mu, 'C1Stick_1_lambda_par': lambda_par}

    # signal computation
    wrapped_signal = stick_model_wrapped(acq_wrapped)
    naked_signal = stick_model.simulate_signal(acq_scheme, parameters)
    # assert list(wrapped_signal) == list(naked_signal)
    np.testing.assert_allclose(wrapped_signal, naked_signal, rtol=0, atol=1e-5)

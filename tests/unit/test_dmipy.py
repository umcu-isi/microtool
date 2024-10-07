import numpy as np
import pytest
from dmipy.data import saved_acquisition_schemes
from dmipy.signal_models import cylinder_models

from microtool.dmipy import convert_dmipy_scheme2diffusion_scheme, convert_diffusion_scheme2dmipy_scheme, \
    DmipyMultiTissueModel


def test_scheme_conversion():
    """
    Since some of the DmipyAcquisitionScheme attributes are strange objects we test here for all float,int or
    numpy array attributes to check if they are approximately equal (up to 1% percent deviation is tolerated).

    Failure of this test means there is something wrong with microtool.dmipy.convert_dmipy_scheme2diffusion_scheme or
    microtool.dmipy.convert_diffusion_scheme2dmipy_scheme
    """
    # going over all attributes that are python native or numpy native and asserting equality
    # Standard acquisition scheme from dmipy.data
    dmipy_scheme = saved_acquisition_schemes.wu_minn_hcp_acquisition_scheme()
    # Microtool scheme wrapper
    microtool_scheme = convert_dmipy_scheme2diffusion_scheme(dmipy_scheme)
    # Microtool scheme converter
    converted_microtool_scheme = convert_diffusion_scheme2dmipy_scheme(microtool_scheme)

    dmipy_attributes = vars(dmipy_scheme)
    converted_attributes = vars(converted_microtool_scheme)
    for attribute, value in dmipy_attributes.items():
        report = f"The attribute {attribute} triggered an assertion, i.e., they are not equal for both " \
                 f"scheme types "
        if isinstance(value, np.ndarray):
            # TODO: isinstance(value, np.ndarray) doesn't work for pint-wrapped arrays.
            np.testing.assert_allclose(value, converted_attributes[attribute], rtol=0.01), report
        elif isinstance(value, (float, int)):
            assert value == converted_attributes[attribute], report


class TestFractions:
    mu = (np.pi / 2., np.pi / 2.)  # in radians
    lambda_par = 1.7e-9  # in m^2/s
    stick1 = cylinder_models.C1Stick(mu=mu, lambda_par=lambda_par)

    lambda_par = 0  # in m^2/s
    stick2 = cylinder_models.C1Stick(mu=mu, lambda_par=lambda_par)

    # Create single- and multi-tissue models.
    vf1 = 0.3
    vf2 = 0.7
    model1 = DmipyMultiTissueModel(stick1)
    model2 = DmipyMultiTissueModel(stick2)
    model12 = DmipyMultiTissueModel([stick1, stick2], [vf1, vf2])

    # Create a diffusion acquisition scheme.
    dmipy_scheme = saved_acquisition_schemes.wu_minn_hcp_acquisition_scheme()
    scheme = convert_dmipy_scheme2diffusion_scheme(dmipy_scheme)

    def test_errors(self):
        # Providing two tissue models without volume fractions should raise a ValueError.
        with pytest.raises(ValueError):
            DmipyMultiTissueModel([self.stick1, self.stick2])

        # Providing volume fractions that don't add up to 1 should raise a ValueError.
        with pytest.raises(ValueError):
            DmipyMultiTissueModel([self.stick1, self.stick2], [0.1, 0.2])

        # Providing fever or more volume fractions than tissue models should raise an error.
        with pytest.raises(ValueError):
            DmipyMultiTissueModel([self.stick1, self.stick2], [0.3, 0.6, 0.1])
        with pytest.raises(ValueError):
            DmipyMultiTissueModel([self.stick1, self.stick2], [1.0])

    def test_signal(self):
        # Simulate signals and check if the signal of the multi-tissue model matches the combined signal of the
        # single-tissue models.
        signal1 = self.model1(self.scheme)
        signal2 = self.model2(self.scheme)
        signal12 = self.model12(self.scheme)
        assert signal12 == pytest.approx(self.vf1 * signal1 + self.vf2 * signal2, rel=1e-6)

    def test_jacobian(self):
        # Check if the Jacobian of the multi-tissue model matches the scaled Jacobians of the single-tissue models.
        jac1 = self.model1.scaled_jacobian(self.scheme)
        jac2 = self.model2.scaled_jacobian(self.scheme)
        jac12 = self.model12.scaled_jacobian(self.scheme)
        assert jac12[:, :3] == pytest.approx(self.vf1 * jac1, rel=1e-6)
        assert jac12[:, 3:6] == pytest.approx(self.vf2 * jac2, rel=1e-6)

        # Check if the number of columns in the Jacobian is correct (2×3 parameters + 2 volume fractions = 8).
        assert jac12.shape[1] == 8

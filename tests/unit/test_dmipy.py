import numpy as np
import pytest
from dmipy_fit.data import saved_acquisition_schemes
from dmipy_fit.signal_models import cylinder_models
from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_bvalues

from microtool.dmipy import convert_dmipy_scheme2diffusion_scheme, convert_diffusion_scheme2dmipy_scheme, \
    DmipyMultiTissueModel

from microtool.constants import B_MAX


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
    dmipy_scheme = _in_bounds_scheme()
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

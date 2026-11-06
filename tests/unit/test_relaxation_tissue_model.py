import numpy as np
import pytest

from microtool.acquisition_scheme import DiffusionAcquisitionScheme
from microtool.constants import BASE_SIGNAL_KEY
from microtool.tissue_model import (
    FittedModel,
    MultiTissueModel,
    RelaxationTissueModel,
    TissueModel,
    TissueParameter,
)
from microtool.utils.unit_registry import unit


class ConstantTissueModel(TissueModel):
    def __init__(self):
        super().__init__(
            {
                "amplitude": TissueParameter(
                    value=1.0, scale=1.0, optimize=True, fit_flag=False
                ),
                "fixed_parameter": TissueParameter(
                    value=1.0, scale=1.0, optimize=False, fit_flag=False
                ),
                BASE_SIGNAL_KEY: TissueParameter(
                    value=1.0, scale=1.0, optimize=False, fit_flag=False
                ),
            }
        )
        self.jacobian_calls = 0

    def __call__(self, scheme):
        return np.full(scheme.pulse_count, self["amplitude"].value)

    def scaled_jacobian(self, scheme):
        self.jacobian_calls += 1
        return np.ones((scheme.pulse_count, 1)) * self["amplitude"].scale

    def fit(self, scheme, signal, **fit_options) -> FittedModel:
        raise NotImplementedError


def _scheme(echo_times):
    count = len(echo_times)
    return DiffusionAcquisitionScheme(
        pulse_magnitudes=np.zeros(count) * unit("mT/mm"),
        gradient_directions=np.tile([1.0, 0.0, 0.0], (count, 1)),
        pulse_durations=np.full(count, 0.01) * unit("s"),
        pulse_intervals=np.full(count, 0.02) * unit("s"),
        echo_times=np.asarray(echo_times) * unit("s"),
    )


def _seconds(value):
    to_seconds = getattr(value, "to", None)
    if to_seconds is not None:
        value = to_seconds("s").magnitude
    return np.asarray(value, dtype=float)


def test_relaxation_scaled_jacobian_uses_wrapped_jacobian_and_analytic_t2():
    wrapped_model = ConstantTissueModel()
    model = RelaxationTissueModel(wrapped_model, t2=25e-3 * unit("s"))
    scheme = _scheme([0.035, 0.04, 0.045, 0.05])

    jacobian = model.scaled_jacobian(scheme)

    echo_times = _seconds(scheme.echo_times)
    t2 = float(_seconds(model["T2"].value))
    decay = np.exp(-echo_times / t2)
    expected_t2_derivative = (
        decay * echo_times / t2**2 * float(_seconds(model["T2"].scale))
    )
    assert wrapped_model.jacobian_calls == 1
    assert jacobian.shape == (scheme.pulse_count, 2)
    np.testing.assert_allclose(jacobian[:, 0], decay)
    np.testing.assert_allclose(jacobian[:, 1], expected_t2_derivative)
    assert np.all(jacobian[:, 1] > 0)


def test_relaxation_fit_estimates_t2_from_full_signal_when_fit_enabled():
    model = RelaxationTissueModel(ConstantTissueModel(), t2=20e-3 * unit("s"))
    model["T2"].fit_flag = True
    true_t2 = 34e-3
    scheme = _scheme([0.035, 0.038, 0.042, 0.046, 0.05])
    signal = np.exp(-_seconds(scheme.echo_times) / true_t2)

    fitted = model.fit(scheme, signal, max_nfev=100)

    assert float(_seconds(fitted.fitted_parameters["T2"])) == pytest.approx(
        true_t2, rel=1e-5
    )


def test_multitissue_relaxation_keeps_distinct_t2_values_and_jacobian_columns():
    first = RelaxationTissueModel(ConstantTissueModel(), t2=20e-3 * unit("s"))
    second = RelaxationTissueModel(ConstantTissueModel(), t2=45e-3 * unit("s"))
    model = MultiTissueModel([first, second], volume_fractions=[0.3, 0.7])
    model["vf_1"].fit_flag = False
    scheme = _scheme([0.035, 0.04, 0.045, 0.05])

    jacobian = model.scaled_jacobian(scheme)
    names = np.asarray(model.parameter_names)[model.include_optimize]
    echo_times = _seconds(scheme.echo_times)

    for compartment, fraction in enumerate(model.volume_fractions):
        t2_name = f"model_{compartment}_T2"
        column = np.flatnonzero(names == t2_name).item()
        relaxed_signal = model._models[compartment](scheme)
        t2 = float(_seconds(model._models[compartment]["T2"].value))
        scale = float(_seconds(model._models[compartment]["T2"].scale))
        expected = fraction * relaxed_signal * echo_times / t2**2 * scale
        np.testing.assert_allclose(jacobian[:, column], expected)

    model.set_scaled_fit_parameters(np.array([25.0, 40.0]))
    assert float(_seconds(first["T2"].value)) == pytest.approx(25e-3)
    assert float(_seconds(second["T2"].value)) == pytest.approx(40e-3)

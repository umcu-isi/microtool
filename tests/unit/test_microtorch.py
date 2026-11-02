import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from microtool.acquisition_scheme import DiffusionAcquisitionScheme
from microtool.microtorch import (
    MicrotorchTissueModel,
    convert_acquisition_scheme,
)


class DummyDiffusionScheme(DiffusionAcquisitionScheme):
    def __init__(self):
        pass

    b_values = property(lambda self: np.array([0.0, 1.0, 2.0]))
    b_vectors = property(lambda self: np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                                               [0.0, 1.0, 0.0]]))
    pulse_magnitude = property(lambda self: np.array([0.0, 0.01, 0.02]))
    pulse_durations = property(lambda self: np.array([0.01, 0.01, 0.01]))
    pulse_intervals = property(lambda self: np.array([0.03, 0.03, 0.03]))
    echo_times = property(lambda self: np.array([0.05, 0.05, 0.05]))
    pulse_count = property(lambda self: 3)


class DummyMicrotorchAcquisitionScheme:
    def __init__(self, **parameters):
        self.__dict__.update(parameters)
        self.number_of_measurements = len(parameters["bvalues"])


class DummyTensor(np.ndarray):
    def __new__(cls, values, dtype=None):
        return np.asarray(values, dtype=dtype).view(cls)

    def __array_finalize__(self, _obj):
        pass

    def requires_grad_(self, _requires_grad=True):
        return self

    def numel(self):
        return self.size

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


class DummyMicrotorchModel:
    parameter_names = ["diffusivity"]
    parameter_ranges = [[0.5, 2.5]]

    def __call__(self, scheme, parameters):
        return DummyTensor(
            np.exp(-scheme.bvalues * parameters[0, 0]).reshape(1, -1)
        )


class DummyTwoParameterModel:
    parameter_names = ["diffusivity", "offset"]
    parameter_ranges = [[0.5, 2.5], [-1.0, 1.0]]

    def __call__(self, scheme, parameters):
        return DummyTensor(
            (
                np.exp(-scheme.bvalues * parameters[0, 0]) + parameters[0, 1]
            ).reshape(1, -1)
        )


class DummyCompositeMicrotorchModel(DummyMicrotorchModel):
    parameter_names = ["diffusivity", "f_0", "f_1", "f_2"]
    parameter_ranges = [[0.5, 2.5]]
    n_parameters = 1
    n_fractions = 3

    def __call__(self, scheme, parameters):
        diffusivity, f_0, f_1, f_2 = parameters[0]
        attenuation = np.exp(-scheme.bvalues * diffusivity)
        return DummyTensor((f_0 * attenuation + f_1 + f_2).reshape(1, -1))


@pytest.fixture
def install_microtorch(monkeypatch):
    torch_module = ModuleType("torch")
    torch_module.Tensor = DummyTensor
    torch_module.float32 = np.float32
    torch_module.as_tensor = DummyTensor

    def jacobian(function, parameters):
        parameters = DummyTensor(parameters)
        input_shape = parameters.shape
        columns = []
        for index in range(parameters.size):
            step = 1e-4
            forward = parameters.copy()
            backward = parameters.copy()
            forward.flat[index] += step
            backward.flat[index] -= step
            columns.append(
                (function(forward).numpy() - function(backward).numpy()) / (2 * step)
            )
        return DummyTensor(np.stack(columns, axis=-1).reshape(
            function(parameters).shape + input_shape
        ))

    torch_module.autograd = SimpleNamespace(
        functional=SimpleNamespace(jacobian=jacobian)
    )
    torch_module.cat = lambda tensors, dim=-1: DummyTensor(
        np.concatenate(tensors, axis=dim)
    )
    torch_module.sum = lambda tensor, dim=None, keepdim=False: DummyTensor(
        np.sum(tensor, axis=dim, keepdims=keepdim)
    )

    microtorch_module = ModuleType("microtorch")
    microtorch_module.__path__ = []
    utils_module = ModuleType("microtorch.utils")
    utils_module.__path__ = []
    acquisition_module = ModuleType("microtorch.utils.acquisition_scheme")
    acquisition_module.AcquisitionScheme = DummyMicrotorchAcquisitionScheme

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "microtorch", microtorch_module)
    monkeypatch.setitem(sys.modules, "microtorch.utils", utils_module)
    monkeypatch.setitem(
        sys.modules, "microtorch.utils.acquisition_scheme", acquisition_module
    )


def test_convert_acquisition_scheme(install_microtorch):
    converted = convert_acquisition_scheme(DummyDiffusionScheme())

    np.testing.assert_array_equal(converted.bvalues, [0.0, 0.001, 0.002])
    np.testing.assert_array_equal(converted.bvecs[1], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(converted.gradient_strengths, [0.0, 0.01, 0.02])
    np.testing.assert_array_equal(converted.delta, [0.01, 0.01, 0.01])
    np.testing.assert_array_equal(converted.Delta, [0.03, 0.03, 0.03])
    np.testing.assert_array_equal(converted.TE, [0.05, 0.05, 0.05])


def test_microtorch_model_generates_signal_and_scaled_jacobian(install_microtorch):
    wrapped = MicrotorchTissueModel(
        DummyMicrotorchModel(),
        parameters={"diffusivity": 1.5},
        s0=2.0,
    )
    scheme = DummyDiffusionScheme()

    microtorch_b_values = scheme.b_values * 1e-3
    np.testing.assert_allclose(
        wrapped(scheme), 2.0 * np.exp(-microtorch_b_values * 1.5)
    )
    expected_jacobian = (
        -2.0
        * microtorch_b_values
        * np.exp(-microtorch_b_values * 1.5)
        * wrapped["diffusivity"].scale
    )
    np.testing.assert_allclose(
        wrapped.scaled_jacobian(scheme)[:, 0], expected_jacobian, rtol=1e-3
    )


def test_scaled_jacobian_includes_only_optimized_parameters(install_microtorch):
    wrapped = MicrotorchTissueModel(DummyTwoParameterModel())
    wrapped["diffusivity"].optimize = False

    jacobian = wrapped.scaled_jacobian(DummyDiffusionScheme())

    assert jacobian.shape == (3, 1)
    assert jacobian[0, 0] == 0.0
    np.testing.assert_allclose(jacobian[1:, 0], [1.0, 1.0], rtol=1e-5)


def test_microtorch_model_fit_uses_microtool_parameters(install_microtorch):
    wrapped = MicrotorchTissueModel(DummyMicrotorchModel())
    scheme = DummyDiffusionScheme()
    expected = 1.8
    signal = np.exp(-(scheme.b_values * 1e-3) * expected)

    fitted = wrapped.fit(scheme, signal, method="Powell")

    assert fitted.fitted_parameters["diffusivity"] == pytest.approx(expected, rel=1e-4)


def test_microtorch_model_rejects_unknown_initial_parameters():
    with pytest.raises(ValueError, match="Unknown MicroTorch parameter"):
        MicrotorchTissueModel(
            DummyMicrotorchModel(),
            parameters={"unknown": 1.0},
        )


def test_duplicate_microtorch_parameter_names_are_disambiguated():
    model = DummyMicrotorchModel()
    model.parameter_names = ["diffusivity", "diffusivity"]
    model.parameter_ranges = [[0.5, 2.5], [1.0, 3.0]]

    wrapped = MicrotorchTissueModel(model)

    assert wrapped.parameter_names == ["diffusivity_0", "diffusivity_1", "S0"]


def test_modelmaker_fraction_ranges_are_added_when_omitted():
    wrapped = MicrotorchTissueModel(
        DummyCompositeMicrotorchModel(),
        parameters={"f_0": 0.6, "f_1": 0.35, "f_2": 0.05},
    )

    assert wrapped["f_0"].fit_bounds == (0.0, 1.0)
    assert wrapped["f_1"].fit_bounds == (0.0, 1.0)
    assert wrapped["f_2"].value == pytest.approx(0.05)
    assert wrapped["f_2"].optimize is False
    assert wrapped["f_2"].fit_flag is False


def test_composition_fractions_stay_normalized_and_b0_is_invariant(install_microtorch):
    wrapped = MicrotorchTissueModel(
        DummyCompositeMicrotorchModel(),
        parameters={"diffusivity": 1.5, "f_0": 0.6, "f_1": 0.35, "f_2": 0.05},
    )
    scheme = DummyDiffusionScheme()
    initial_signal = wrapped(scheme)

    scaled_parameters = wrapped.scaled_parameter_vector.copy()
    scaled_parameters[1] += 0.01
    wrapped.set_scaled_parameters(scaled_parameters)
    updated_signal = wrapped(scheme)

    assert wrapped["f_0"].value + wrapped["f_1"].value + wrapped["f_2"].value == pytest.approx(1.0)
    assert wrapped["f_2"].value == pytest.approx(0.04)
    assert updated_signal[0] == initial_signal[0] == pytest.approx(1.0)
    assert updated_signal[1] != pytest.approx(initial_signal[1])


def test_composition_jacobian_has_zero_b0_and_dependent_fraction_column(install_microtorch):
    wrapped = MicrotorchTissueModel(
        DummyCompositeMicrotorchModel(),
        parameters={"diffusivity": 1.5, "f_0": 0.6, "f_1": 0.35, "f_2": 0.05},
    )

    jacobian = wrapped.scaled_jacobian(DummyDiffusionScheme())

    assert jacobian.shape == (3, 3)
    np.testing.assert_array_equal(jacobian[0], np.zeros(3))
    expected_f0_derivative = np.exp(-np.array([0.001, 0.002]) * 1.5) - 1.0
    np.testing.assert_allclose(jacobian[1:, 1], expected_f0_derivative, rtol=1e-3)


def test_composition_rejects_negative_derived_fraction():
    wrapped = MicrotorchTissueModel(
        DummyCompositeMicrotorchModel(),
        parameters={"f_0": 0.6, "f_1": 0.35, "f_2": 0.05},
    )
    scaled_parameters = wrapped.scaled_parameter_vector.copy()
    scaled_parameters[1:3] = [0.7, 0.35]

    with pytest.raises(ValueError, match="must sum to at most 1"):
        wrapped.set_scaled_parameters(scaled_parameters)

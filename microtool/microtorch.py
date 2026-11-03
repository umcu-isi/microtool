"""Adapters for using MicroTorch models with MICROtool."""
from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.optimize import NonlinearConstraint, minimize

from .acquisition_scheme import (
    DiffusionAcquisitionScheme,
    DiffusionAcquisitionSchemeBValue,
    DiffusionAcquisitionSchemeDelta,
)
from .constants import BASE_SIGNAL_KEY
from .tissue_model import (
    FittedModelMinimize,
    TissueModel,
    TissueParameter,
)
from .utils.unit_registry import cast_to_ndarray


_SUPPORTED_SCHEMES = (
    DiffusionAcquisitionScheme,
    DiffusionAcquisitionSchemeBValue,
    DiffusionAcquisitionSchemeDelta,
)


def _microtorch_api():
    try:
        import torch
        from microtorch.utils.acquisition_scheme import AcquisitionScheme
    except ImportError as exc:
        raise ImportError(
            "MicroTorch support requires the 'microtorch-mri' package and PyTorch. "
            "Install the MICROtool MicroTorch extra to use this adapter."
        ) from exc
    return torch, AcquisitionScheme


def convert_acquisition_scheme(scheme: DiffusionAcquisitionScheme):
    """Convert a MICROtool diffusion scheme to MicroTorch's acquisition scheme."""
    if not isinstance(scheme, _SUPPORTED_SCHEMES):
        raise TypeError(
            f"scheme is of type {type(scheme)}, expected a diffusion acquisition scheme."
        )
    
    #Microtorch uses b-values in units ms/mu^2, so that is manually converted 
    bvalues = cast_to_ndarray(scheme.b_values, unit_str="s/mm²") * 1e-3


    bvecs = cast_to_ndarray(scheme.b_vectors, unit_str="dimensionless")
    gradient_strengths = cast_to_ndarray(scheme.pulse_magnitude, unit_str="T/m")
    
    #Microtorch uses delta and Delta in units of ms, so that is manually converted
    delta = cast_to_ndarray(scheme.pulse_durations, unit_str="s") * 1e3
    Delta = cast_to_ndarray(scheme.pulse_intervals, unit_str="s") * 1e3
    TE = cast_to_ndarray(scheme.echo_times, unit_str="s")

    _, microtorch_scheme = _microtorch_api()
    return microtorch_scheme(
        bvalues=bvalues,
        bvecs=bvecs,
        gradient_strengths=gradient_strengths,
        delta=delta,
        Delta=Delta,
        TE=TE,
    )

class MicrotorchTissueModel(TissueModel):
    """Wrap a MicroTorch compartment or ``ModelMaker`` model for MICROtool.

    ``model`` must expose ``parameter_names``, ``parameter_ranges`` and be
    callable as ``model(acquisition_scheme, parameter_tensor)``. Parameter
    values may be supplied by name; otherwise each parameter starts at the
    midpoint of its MicroTorch range.
    """

    def __init__(
        self,
        model,
        parameters: Optional[Dict[str, float]] = None,
        s0: float = 1.0,
    ):
        if not callable(model):
            raise TypeError("model must be a callable MicroTorch model.")
        if not hasattr(model, "parameter_names") or not hasattr(model, "parameter_ranges"):
            raise TypeError(
                "model must expose MicroTorch 'parameter_names' and 'parameter_ranges'."
            )

        model_parameter_names = list(model.parameter_names)
        parameter_ranges = np.asarray(model.parameter_ranges, dtype=float)
        fraction_count = getattr(model, "n_fractions", 0)
        compartment_parameter_count = getattr(model, "n_parameters", len(model_parameter_names))
        fraction_names = [f"f_{index}" for index in range(fraction_count)]
        has_model_fractions = (
            fraction_count > 1
            and len(model_parameter_names) == compartment_parameter_count + fraction_count
            and model_parameter_names[-fraction_count:] == fraction_names
        )
        if (
            fraction_count
            and len(model_parameter_names) == compartment_parameter_count + fraction_count
            and parameter_ranges.shape == (compartment_parameter_count, 2)
            and model_parameter_names[-fraction_count:]
            == [f"f_{index}" for index in range(fraction_count)]
        ):
            parameter_ranges = np.vstack(
                (parameter_ranges, np.tile((0.0, 1.0), (fraction_count, 1)))
            )
        if not model_parameter_names and parameter_ranges.size == 0:
            parameter_ranges = parameter_ranges.reshape((0, 2))
        if parameter_ranges.shape != (len(model_parameter_names), 2):
            raise ValueError(
                "MicroTorch parameter_ranges must contain one lower/upper bound pair "
                "for each parameter_name."
            )

        microtorch_names = self._unique_parameter_names(model_parameter_names)
        dependent_fraction_name = fraction_names[-1] if has_model_fractions else None
        if BASE_SIGNAL_KEY in microtorch_names:
            raise ValueError(
                f"MicroTorch parameter names cannot include the reserved name {BASE_SIGNAL_KEY!r}."
            )
        supplied_parameters = parameters or {}
        unknown_parameters = set(supplied_parameters).difference(microtorch_names)
        if unknown_parameters:
            raise ValueError(
                f"Unknown MicroTorch parameter name(s): {', '.join(sorted(unknown_parameters))}."
            )

        wrapped_parameters = {}
        for name, (lower, upper) in zip(microtorch_names, parameter_ranges):
            if np.isnan(lower) or np.isnan(upper) or lower > upper:
                raise ValueError(f"Invalid parameter range for {name}: ({lower}, {upper}).")

            if name in supplied_parameters:
                value = float(supplied_parameters[name])
            elif name.startswith("f_") and fraction_count:
                equal_fraction = 1.0 / fraction_count
                value = (
                    equal_fraction
                    if lower <= equal_fraction <= upper
                    else float((lower + upper) / 2.0)
                )
            elif np.isfinite(lower) and np.isfinite(upper):
                value = float((lower + upper) / 2.0)
            elif np.isfinite(lower):
                value = float(lower + max(abs(lower), 1.0))
            elif np.isfinite(upper):
                value = float(upper - max(abs(upper), 1.0))
            else:
                value = 1.0

            if not np.isfinite(value) or value < lower or value > upper:
                raise ValueError(
                    f"Initial value for {name} must be finite and within "
                    f"({lower}, {upper}); received {value}."
                )

            finite_bounds = [abs(bound) for bound in (lower, upper) if np.isfinite(bound)]
            scale = 1
            is_fixed = lower == upper
            parameter = TissueParameter(
                value=value,
                scale=scale,
                optimize=not is_fixed and name != dependent_fraction_name,
                fit_flag=not is_fixed and name != dependent_fraction_name,
                fit_bounds=(float(lower), float(upper)),
            )
            parameter.fit_guess = value
            wrapped_parameters[name] = parameter

        wrapped_parameters[BASE_SIGNAL_KEY] = TissueParameter(
            value=float(s0),
            scale=1.0,
            optimize=False,
            fit_flag=False,
            fit_bounds=(0.0, 2.0),
        )
        self._model = model
        self._model_parameter_names = microtorch_names
        self._fraction_parameter_names = fraction_names if has_model_fractions else []
        self._fraction_start = compartment_parameter_count
        self._dependent_fraction_name = dependent_fraction_name
        super().__init__(wrapped_parameters)
        if self._dependent_fraction_name is not None:
            requested_dependent_fraction = supplied_parameters.get(dependent_fraction_name)
            self._synchronize_dependent_fraction()
            if requested_dependent_fraction is not None and not np.isclose(
                float(requested_dependent_fraction),
                self[dependent_fraction_name].value,
                rtol=0.0,
                atol=1e-8,
            ):
                raise ValueError(
                    "MicroTorch compartment volume fractions must sum to 1. "
                    f"{dependent_fraction_name} is derived from the other fractions."
                )

    @property
    def microtorch_model(self):
        """Return the wrapped MicroTorch model."""
        return self._model

    @staticmethod
    def _unique_parameter_names(names: Sequence[str]) -> list[str]:
        totals = {name: names.count(name) for name in set(names)}
        seen = {}
        unique_names = []
        for name in names:
            if totals[name] == 1:
                unique_names.append(name)
                continue
            index = seen.get(name, 0)
            unique_names.append(f"{name}_{index}")
            seen[name] = index + 1
        return unique_names

    def _predict_signal(
        self,
        scheme: DiffusionAcquisitionScheme,
        *,
        validate_composition: bool,
    ) -> np.ndarray:
        self._synchronize_dependent_fraction(validate=validate_composition)
        microtorch_scheme = convert_acquisition_scheme(scheme)
        torch, _ = _microtorch_api()
        model_parameters = np.asarray(
            [self[name].value for name in self._model_parameter_names],
            dtype=np.float32,
        ).reshape(1, -1)
        parameter_tensor = torch.as_tensor(model_parameters, dtype=torch.float32)
        signal = self._model(microtorch_scheme, parameter_tensor)

        if hasattr(signal, "detach"):
            signal = signal.detach().cpu().numpy()
        signal = np.asarray(signal, dtype=float)
        if signal.size != scheme.pulse_count:
            raise ValueError(
                f"MicroTorch returned {signal.size} signal values for "
                f"{scheme.pulse_count} measurements."
            )
        signal = signal.reshape(-1)
        b0_mask = cast_to_ndarray(scheme.b_values, unit_str="s/mm²") == 0
        signal[b0_mask] = 1.0
        return self[BASE_SIGNAL_KEY].value * signal

    def __call__(self, scheme: DiffusionAcquisitionScheme) -> np.ndarray:
        return self._predict_signal(scheme, validate_composition=True)

    def scaled_jacobian(self, scheme: DiffusionAcquisitionScheme) -> np.ndarray:
        """Return signal derivatives for parameters marked for optimization.

        Derivatives are taken with respect to the MicroTorch parameter tensor,
        then multiplied by MICROtool parameter scales to produce the scaled
        Jacobian expected by ``TissueModel``.
        """
        self._synchronize_dependent_fraction()
        microtorch_scheme = convert_acquisition_scheme(scheme)
        torch, _ = _microtorch_api()
        parameter_values = np.asarray(
            [self[name].value for name in self._model_parameter_names],
            dtype=np.float32,
        ).reshape(1, -1)
        parameter_tensor = torch.as_tensor(
            parameter_values, dtype=torch.float32
        ).requires_grad_(True)

        def signal_from_parameters(parameters):
            model_parameters = parameters
            if self._dependent_fraction_name is not None:
                dependent_index = self._model_parameter_names.index(
                    self._dependent_fraction_name
                )
                dependent_value = 1.0 - torch.sum(
                    parameters[
                        ...,
                        self._fraction_start:dependent_index,
                    ],
                    dim=-1,
                    keepdim=True,
                )
                model_parameters = torch.cat(
                    (parameters[..., :dependent_index], dependent_value),
                    dim=-1,
                )

            signal = self._model(microtorch_scheme, model_parameters)
            if signal.numel() != scheme.pulse_count:
                raise ValueError(
                    f"MicroTorch returned {signal.numel()} signal values for "
                    f"{scheme.pulse_count} measurements."
                )
            return self[BASE_SIGNAL_KEY].value * signal.reshape(-1)

        jacobian = torch.autograd.functional.jacobian(
            signal_from_parameters,
            parameter_tensor,
        )
        jacobian = jacobian.reshape(scheme.pulse_count, len(self._model_parameter_names))
        jacobian = jacobian.detach().cpu().numpy()

        scales = np.asarray(
            [self[name].scale for name in self._model_parameter_names],
            dtype=float,
        )
        optimize_mask = np.asarray(
            [self[name].optimize for name in self._model_parameter_names],
            dtype=bool,
        )
        jacobian = (jacobian * scales[np.newaxis, :])[:, optimize_mask]
        b0_mask = cast_to_ndarray(scheme.b_values, unit_str="s/mm²") == 0
        jacobian[b0_mask, :] = 0.0
        return jacobian

    def _synchronize_dependent_fraction(self, *, validate: bool = True) -> None:
        if self._dependent_fraction_name is None:
            return

        dependent = self[self._dependent_fraction_name]
        fraction_sum = sum(
            self[name].value
            for name in self._fraction_parameter_names
            if name != self._dependent_fraction_name
        )
        value = 1.0 - fraction_sum
        lower, upper = dependent.fit_bounds
        if validate and (value < lower - 1e-8 or value > upper + 1e-8):
            raise ValueError(
                "The optimized MicroTorch volume fractions must sum to at most 1; "
                f"the derived {self._dependent_fraction_name} fraction would be {value}."
            )
        dependent.value = min(max(value, lower), upper) if validate else value

    def set_scaled_parameters(self, new_parameter_values: Sequence[float]) -> None:
        if len(new_parameter_values) != len(self):
            raise ValueError(
                f"Expected {len(self)} parameter values, received {len(new_parameter_values)}."
            )
        super().set_scaled_parameters(new_parameter_values)
        self._synchronize_dependent_fraction()

    def set_scaled_fit_parameters(self, new_values: np.ndarray) -> None:
        super().set_scaled_fit_parameters(new_values)
        self._synchronize_dependent_fraction()

    def _set_scaled_fit_parameters_for_fit(self, new_values: np.ndarray) -> None:
        super().set_scaled_fit_parameters(new_values)
        self._synchronize_dependent_fraction(validate=False)

    def fit(
        self,
        scheme: DiffusionAcquisitionScheme,
        signal: np.ndarray,
        method: str = "trust-constr",
        **fit_options,
    ) -> FittedModelMinimize:
        if not isinstance(scheme, _SUPPORTED_SCHEMES):
            raise TypeError("MicrotorchTissueModel.fit requires a diffusion acquisition scheme.")

        signal = np.asarray(signal, dtype=float)
        if signal.shape != (scheme.pulse_count,):
            raise ValueError("signal must have one value for each measurement.")
        if not np.any(self.include_fit):
            raise ValueError("At least one MicroTorch model parameter must be fit.")
        if (
            self._dependent_fraction_name is not None
            and self[self._dependent_fraction_name].fit_flag
        ):
            raise ValueError(
                f"{self._dependent_fraction_name} is derived from the other "
                "volume fractions and cannot be fit independently; set its "
                "fit_flag to False."
            )

        minimize_options = dict(fit_options)
        constraints = list(minimize_options.pop("constraints", ()))
        if self._fraction_parameter_names:
            fitted_names = [
                name for name, parameter in self.items() if parameter.fit_flag
            ]
            fraction_indices = [
                index
                for index, name in enumerate(fitted_names)
                if name in self._fraction_parameter_names
            ]
            dependent = self[self._dependent_fraction_name]
            fixed_fraction_sum = sum(
                self[name].value
                for name in self._fraction_parameter_names
                if name != self._dependent_fraction_name
                and name not in fitted_names
            )
            fraction_scales = np.asarray(
                [self[fitted_names[index]].scale for index in fraction_indices],
                dtype=float,
            )
            constraints.append(
                NonlinearConstraint(
                    lambda values: (
                        np.dot(values[fraction_indices], fraction_scales)
                        + fixed_fraction_sum
                    ),
                    1.0 - dependent.fit_bounds[1],
                    1.0 - dependent.fit_bounds[0],
                )
            )

        def objective(fit_parameter_vector, observed_signal, fit_scheme, model):
            model._set_scaled_fit_parameters_for_fit(fit_parameter_vector)
            predicted_signal = model._predict_signal(
                fit_scheme,
                validate_composition=False,
            )
            square_diff = (observed_signal - predicted_signal) ** 2
            return np.sum(square_diff)

        result = minimize(
            objective,
            x0=self.scaled_fit_initial_guess,
            args=(signal, scheme, deepcopy(self)),
            bounds=self.scaled_fit_bounds_all,
            method=method,
            constraints=constraints,
            **minimize_options,
        )
        return FittedModelMinimize(self, result)


def make_microtool_tissue_model(
    model,
    parameters: Optional[Dict[str, float]] = None,
    s0: float = 1.0,
) -> MicrotorchTissueModel:
    """Create a MICROtool tissue model from a MicroTorch model."""
    return MicrotorchTissueModel(model, parameters=parameters, s0=s0)

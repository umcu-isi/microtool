"""Adapters for using MicroTorch models with MICROtool."""
from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.optimize import NonlinearConstraint, minimize, least_squares
from copy import deepcopy


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
        method: str = "trf",
        max_nfev: int = 150,
        **fit_options,
    ) -> FittedModelMinimize:


        if not isinstance(scheme, _SUPPORTED_SCHEMES):
            raise TypeError(
                "MicroTorchTissueModel.fit requires a diffusion acquisition scheme."
            )

        signal = np.asarray(signal, dtype=float).ravel()

        if signal.shape != (scheme.pulse_count,):
            raise ValueError(
                "signal must have one value for each measurement."
            )

        if not np.any(self.include_fit):
            raise ValueError(
                "At least one MicroTorch model parameter must be fit."
            )

        if (
            self._dependent_fraction_name is not None
            and self[self._dependent_fraction_name].fit_flag
        ):
            raise ValueError(
                f"{self._dependent_fraction_name} is derived from the other "
                "volume fractions and cannot be fit independently; set its "
                "fit_flag to False."
            )

        # Get fitted parameter names
        fit_names = [
            name for name, parameter in self.items()
            if parameter.fit_flag
        ]

        # Parameter scaling
        scales = np.array(
            [self[name].scale for name in fit_names],
            dtype=float,
        )

        initial_values = np.array(
            [self[name].value for name in fit_names],
            dtype=float,
        )

        lower = np.array(
            [self[name].fit_bounds[0] for name in fit_names],
            dtype=float,
        )

        upper = np.array(
            [self[name].fit_bounds[1] for name in fit_names],
            dtype=float,
        )

        if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
            raise ValueError(
                "All fitted parameter scales must be positive."
            )

        # Normalize parameters
        x0 = initial_values / scales
        bounds = (lower / scales, upper / scales)

        fraction_fit_names = [
            name for name in self._fraction_parameter_names
            if self[name].fit_flag
        ]
        fraction_indices = {
            name: fit_names.index(name) for name in fraction_fit_names
        }
        fraction_sum_lower = None
        fraction_sum_upper = None
        fraction_fixed_sum = 0.0
        if fraction_fit_names:
            dependent = self[self._dependent_fraction_name]
            dependent_lower, dependent_upper = dependent.fit_bounds
            fixed_fraction_names = [
                name for name in self._fraction_parameter_names
                if name != self._dependent_fraction_name
                and name not in fraction_fit_names
            ]
            fraction_fixed_sum = sum(
                self[name].value for name in fixed_fraction_names
            )
            fraction_sum_lower = max(
                sum(self[name].fit_bounds[0] for name in fraction_fit_names),
                1.0 - fraction_fixed_sum - dependent_upper,
            )
            fraction_sum_upper = min(
                sum(self[name].fit_bounds[1] for name in fraction_fit_names),
                1.0 - fraction_fixed_sum - dependent_lower,
            )
            if fraction_sum_lower > fraction_sum_upper:
                raise ValueError(
                    "MicroTorch fraction bounds have no feasible composition."
                )

            if any(
                not np.isfinite(bound)
                for name in fraction_fit_names
                for bound in self[name].fit_bounds
            ):
                raise ValueError(
                    "Fitted MicroTorch fraction bounds must be finite."
                )

            def feasible_fraction_interval(name, assigned_sum, remaining_names):
                lower_bound, upper_bound = self[name].fit_bounds
                remaining_lower = sum(
                    self[remaining_name].fit_bounds[0]
                    for remaining_name in remaining_names
                )
                remaining_upper = sum(
                    self[remaining_name].fit_bounds[1]
                    for remaining_name in remaining_names
                )
                lower_bound = max(
                    lower_bound,
                    fraction_sum_lower - assigned_sum - remaining_upper,
                )
                upper_bound = min(
                    upper_bound,
                    fraction_sum_upper - assigned_sum - remaining_lower,
                )
                if lower_bound > upper_bound + 1e-12:
                    raise ValueError(
                        "MicroTorch fraction bounds have no feasible composition."
                    )
                if lower_bound > upper_bound:
                    upper_bound = lower_bound
                return lower_bound, upper_bound

            # Represent fitted fractions as bounded latent variables so every
            # least-squares trial remains inside the valid simplex.
            assigned_sum = 0.0
            for index, name in enumerate(fraction_fit_names):
                remaining_names = fraction_fit_names[index + 1 :]
                fraction_lower, fraction_upper = feasible_fraction_interval(
                    name, assigned_sum, remaining_names
                )
                initial_fraction = self[name].value
                if not (
                    fraction_lower - 1e-12
                    <= initial_fraction
                    <= fraction_upper + 1e-12
                ):
                    raise ValueError(
                        f"Initial value for {name} is outside the feasible "
                        "fraction composition."
                    )
                span = fraction_upper - fraction_lower
                x0[fraction_indices[name]] = (
                    0.0
                    if span <= 1e-12
                    else np.clip(
                        (initial_fraction - fraction_lower) / span,
                        0.0,
                        1.0,
                    )
                )
                assigned_sum += initial_fraction
                bounds[0][fraction_indices[name]] = 0.0
                bounds[1][fraction_indices[name]] = 1.0

        # Create an independent model for optimization
        optimization_model = deepcopy(self)

        def decode_parameters(x):
            parameter_values = {
                name: float(value * scale)
                for name, value, scale in zip(fit_names, x, scales)
                if name not in fraction_indices
            }
            assigned_sum = 0.0
            for index, name in enumerate(fraction_fit_names):
                remaining_names = fraction_fit_names[index + 1 :]
                fraction_lower, fraction_upper = feasible_fraction_interval(
                    name, assigned_sum, remaining_names
                )
                latent_value = np.clip(x[fraction_indices[name]], 0.0, 1.0)
                parameter_values[name] = float(
                    fraction_lower
                    + latent_value * (fraction_upper - fraction_lower)
                )
                assigned_sum += parameter_values[name]
            return parameter_values

        def residual_fn(x):
            for name, value in decode_parameters(x).items():
                optimization_model[name].value = value

            # Evaluate forward model
            predicted = np.asarray(
                optimization_model._predict_signal(
                    scheme,
                    validate_composition=True,
                ),
                dtype=float,
            ).ravel()

            if predicted.shape != signal.shape:
                raise ValueError(
                    f"Predicted signal shape {predicted.shape} does not "
                    f"match measured signal shape {signal.shape}."
                )

            return predicted - signal

        # Run nonlinear least squares
        result = least_squares(
            residual_fn,
            x0=x0,
            bounds=bounds,
            method=method,
            max_nfev=max_nfev,
            ftol=1e-5,
            xtol=1e-5,
            verbose=2,
            **fit_options,
        )

        # Construct fitted model
        fitted = deepcopy(self)

        for name, value in decode_parameters(result.x).items():
            fitted[name].value = value
        fitted._synchronize_dependent_fraction()

        return _FittedMicrotorchModel(fitted, result)


class _FittedMicrotorchModel(FittedModelMinimize):
    @property
    def fitted_parameters(self) -> Dict[str, np.ndarray]:
        return {
            name: parameter.value
            for name, parameter in self.model.items()
            if parameter.fit_flag
        }

    @property
    def scaled_fitted_parameters(self) -> np.ndarray:
        return np.asarray(
            [
                parameter.value / parameter.scale
                for parameter in self.model.values()
                if parameter.fit_flag
            ],
            dtype=float,
        )


def make_microtool_tissue_model(
    model,
    parameters: Optional[Dict[str, float]] = None,
    s0: float = 1.0,
) -> MicrotorchTissueModel:
    """Create a MICROtool tissue model from a MicroTorch model."""
    return MicrotorchTissueModel(model, parameters=parameters, s0=s0)

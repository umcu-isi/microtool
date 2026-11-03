import os.path
from typing import Optional, Dict, Union

import numpy as np

from .acquisition_scheme import DiffusionAcquisitionScheme
from .tissue_model import FittedModelMinimize, TissueModel, TissueParameter
from scipy.optimize import least_squares
from copy import deepcopy
import time

matlab_engine_help = 'https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html'

# Try to import the MATLAB engine API for Python and raise a helpful error when it was not found.
try:
    import matlab
    import matlab.engine
    from matlab import double  # Prevents reference errors
except ImportError:
    raise (ImportError(
        'MISST models require the MATLAB engine API for Python, which was not found. Install it as described in: '
        f'{matlab_engine_help}')
    ) from None

_matlab_engine: Optional[matlab.engine.MatlabEngine] = None


# https://nl.mathworks.com/help/matlab/matlab_external/get-started-with-matlab-engine-for-python.html
# syntax: ret = MatlabEngine.matlabfunc(*args, nargout=1, background=False, stdout=sys.stsdout, stderr=sys.stderr)
def _get_matlab_engine():
    global _matlab_engine
    if not _matlab_engine:
        print('Starting MATLAB engine...')
        _matlab_engine = matlab.engine.start_matlab()
        version = _matlab_engine.version('-release')
        print(f'MATLAB {version} started.')
    return _matlab_engine


def set_misst_path(path: str):
    """
    Sets the path to the Microstructure Imaging Sequence Simulation ToolBox.
    Get it from: https://www.nitrc.org/projects/misst

    :param path: Location of the MISST package.
    :raises ValueError: The given path does not exist.
    """
    if os.path.isdir(path):
        engine = _get_matlab_engine()
        all_paths = engine.genpath(path)
        engine.addpath(all_paths)
    else:
        raise ValueError('The given path does not exist.')


def convert_acquisition_scheme(scheme):
    seq_name = scheme.name

    g_vals = (
        scheme["DiffusionPulseMagnitude"]
        .values.astype(float).ravel().tolist()
    )

    smalldel_array = np.asarray(scheme.pulse_durations, dtype=float).ravel()
    smalldel_vals = smalldel_array.tolist()

    delta_vals = (
        scheme["DiffusionPulseInterval"]
        .values.astype(float).ravel().tolist()
    )

    grad_dirs = scheme.b_vectors.astype(float).tolist()

    m_meas = len(g_vals)

    protocol = {
        "pulseseq": "TWOGSE" if seq_name == "OGSE" else seq_name,
        "G": double(g_vals),
        "grad_dirs": double(grad_dirs),
        "smalldel": double(smalldel_vals),
        "delta": double(delta_vals),
        "tau": 1e-4,
    }

    if seq_name == "OGSE":
        # Derive frequency from N and pulse duration.
        n_periods = np.asarray(scheme.n_periods, dtype=float).ravel()

        if len(n_periods) != m_meas:
            raise ValueError(
                "n_periods must have one value per measurement."
            )

        if len(smalldel_array) != m_meas:
            raise ValueError(
                "pulse_durations must have one value per measurement."
            )

        # Avoid division by zero for b0 measurements.
        freq = np.zeros(m_meas, dtype=float)
        valid = smalldel_array > 0
        freq[valid] = n_periods[valid] / smalldel_array[valid]

        omega = 2.0 * np.pi * freq

        protocol["omega"] = double(omega.tolist())
        protocol["slew_rate"] = double([9200.0] * m_meas)
        protocol["apodisedcos"] = 1
        protocol["mirror"] = 0

    return protocol

class MisstTissueModel(TissueModel):
    def __init__(self, name: str, parameters: Dict[str, TissueParameter]):
        super().__init__(parameters)

        # Add the given tissue parameters and S0.
        self.update({**parameters, 'S0': TissueParameter(value=1.0, scale=1.0, optimize=False)})

        # A MISST tissue model is just a MATLAB struct with a name and an array of parameters.
        self._model = {
            'name': name,
            'params': double([p.value for p in parameters.values()])
        }

    def __call__(self, scheme: DiffusionAcquisitionScheme) -> np.ndarray:
        engine = _get_matlab_engine()
        misst_scheme = convert_acquisition_scheme(scheme)

        # Create the 'GEN' protocol (see MISST documentation).
        protocol = {
            'pulseseq': 'GEN',
            'G': engine.wave_form(misst_scheme),
            'tau': misst_scheme['tau'],
        }

        if misst_scheme['pulseseq'] != 'PGSE':
            # Add the matrices and other constants necessary for the matrix method (see MISST documentation).
            # This is not required for pulsed-gradient spin-echo.
            protocol = engine.MMConstants(self._model, protocol)

        # Evaluate the MISST model.
        s0 = self['S0'].value
        return s0 * np.array(engine.SynthMeas(self._model, protocol)).ravel()

    def scaled_jacobian(self, scheme: DiffusionAcquisitionScheme) -> np.ndarray:
        engine = _get_matlab_engine()
        misst_scheme = convert_acquisition_scheme(scheme)

        # Create the GEN protocol.
        protocol = {
            'pulseseq': 'GEN',
            'G': engine.wave_form(misst_scheme),
            'tau': misst_scheme['tau'],
        }

        if misst_scheme['pulseseq'] != 'PGSE':
            protocol = engine.MMConstants(self._model, protocol)

        # Evaluate signal and Jacobian.
        s0 = self['S0'].value
        signal, jac = engine.SynthMeas(self._model, protocol, nargout=2)

        jac = np.asarray(jac, dtype=float)
        signal = np.asarray(signal, dtype=float).reshape(-1, 1)

        # Jacobian columns: MISST parameters, followed by S0.
        jac_full = np.concatenate([s0 * jac, signal], axis=1)

        # Scale each column by its corresponding parameter scale.
        parameter_names = [name for name in self if name != 'S0'] + ['S0']
        scales = np.array(
            [self[name].scale for name in parameter_names],
            dtype=float,
        )

        if jac_full.shape[1] != len(scales):
            raise ValueError(
                f"Jacobian has {jac_full.shape[1]} columns, "
                f"but {len(scales)} parameter scales were provided."
            )

        jac_full *= scales[np.newaxis, :]

        return jac_full[:, self.include_optimize]
    def fit(
            self,
            scheme: DiffusionAcquisitionScheme,
            signal: np.ndarray,
            method: str = "trf",
            max_nfev: int = 150,
            **fit_options,
        ) -> FittedModelMinimize:

            signal = np.asarray(signal, dtype=float).ravel()

            fit_names = [name for name, parameter in self.items() if parameter.fit_flag]
            if not fit_names:
                raise ValueError("No parameters have fit_flag=True.")

            scales = np.array([self[name].scale for name in fit_names], dtype=float)
            initial_values = np.array([self[name].value for name in fit_names], dtype=float)
            lower = np.array([self[name].fit_bounds[0] for name in fit_names], dtype=float)
            upper = np.array([self[name].fit_bounds[1] for name in fit_names], dtype=float)

            if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
                raise ValueError("All fitted parameter scales must be positive.")

            x0 = initial_values / scales
            bounds = (lower / scales, upper / scales)

            optimization_model = deepcopy(self)
            misst_parameter_names = [name for name in optimization_model if name != "S0"]

            def residual_fn(x):
                for name, value, scale in zip(fit_names, x, scales):
                    optimization_model[name].value = float(value * scale)

                if hasattr(optimization_model, "_model") and "_model" in optimization_model.__dict__:
                    optimization_model._model["params"] = double(
                        [optimization_model[name].value for name in misst_parameter_names]
                    )

                predicted = np.asarray(optimization_model(scheme), dtype=float).ravel()

                if predicted.shape != signal.shape:
                    raise ValueError(
                        f"Predicted signal shape {predicted.shape} does not match measured signal shape {signal.shape}."
                    )

                return predicted - signal

            start_time = time.time()

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

            fitted = optimization_model
            for name, value, scale in zip(fit_names, result.x, scales):
                fitted[name].value = float(value * scale)

            if hasattr(fitted, "_model") and "_model" in fitted.__dict__:
                fitted._model["params"] = double(
                    [fitted[name].value for name in misst_parameter_names]
                )

            return FittedModelMinimize(fitted, result)

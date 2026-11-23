import os.path
from typing import Optional, Dict, Union

import numpy as np

from .acquisition_scheme import DiffusionAcquisitionScheme, DiffusionAcquisitionScheme_OGSE
from .tissue_model import TissueModel, TissueParameter
from .scanner_parameters import ScannerParameters

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


def convert_acquisition_scheme(
    scheme: Union[DiffusionAcquisitionScheme, DiffusionAcquisitionScheme_OGSE],
) -> Dict[str, Union[str, double, float, int]]:
    """Convert a diffusion acquisition scheme into a MATLAB-compatible protocol dictionary."""
    seq_name = scheme.name

    # Extract core gradient parameters as flat Python lists
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

    # Assemble base protocol dictionary for MATLAB engine
    protocol = {
        "pulseseq": "TWOGSE" if seq_name == "OGSE" else seq_name,
        "G": double(g_vals),
        "grad_dirs": double(grad_dirs),
        "smalldel": double(smalldel_vals),
        "delta": double(delta_vals),
        "tau": 1e-4,
    }

    if seq_name == "OGSE":
        n_periods = np.asarray(scheme.n_periods, dtype=float).ravel()

        if len(n_periods) != m_meas:
            raise ValueError(
                "n_periods must have one value per measurement."
            )

        if len(smalldel_array) != m_meas:
            raise ValueError(
                "pulse_durations must have one value per measurement."
            )

        # Compute angular frequency (rad/s), leaving b0 measurements at 0
        freq = np.zeros(m_meas, dtype=float)
        valid = smalldel_array > 0
        freq[valid] = n_periods[valid] / smalldel_array[valid]
        omega = 2.0 * np.pi * freq

        # Add trapezoidal-wave OGSE (TWOGSE) parameters
        protocol["omega"] = double(omega.tolist())
        protocol["slew_rate"] = double([ScannerParameters.s_max] * m_meas)
        protocol["apodisedcos"] = 1
        protocol["mirror"] = 0

    return protocol

class MisstTissueModel(TissueModel):
    def __init__(self, name: str, parameters: Dict[str, TissueParameter]):
        super().__init__()

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

        # Evaluate the MISST model and obtain the Jacobian (the SynthMeas function returns the Jacobian as the 2nd
        # output argument).
        s0 = self['S0'].value
        signal, jac = engine.SynthMeas(self._model, protocol, nargout=2)

        # Add the derivative to S0 (the signal itself).
        # Note: parameter scales are all 1.0
        return np.concatenate([s0 * np.array(jac), signal], axis=1)[:, self.include_optimize]

    # TODO: Implement fit method

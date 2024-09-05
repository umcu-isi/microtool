import os.path
from typing import Optional, Dict, Union

import numpy as np

from .acquisition_scheme import DiffusionAcquisitionScheme
from .tissue_model import TissueModel, TissueParameter

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


def convert_acquisition_scheme(scheme: DiffusionAcquisitionScheme) -> Dict[str, Union[str, double, float]]:
    # Create a MISST acquisition scheme.
    return {
        'pulseseq': 'PGSE',  # Standard pulsed-gradient spin-echo. TODO: support more through DiffusionAcquisitionScheme
        'G': double((scheme.pulse_magnitude * 1e-3).tolist()),  # Convert from mT/m to T/m.
        'grad_dirs': double(scheme.b_vectors.tolist()),
        'smalldel': double((scheme.pulse_widths * 1e-3).tolist()),  # Convert from ms to s.
        'delta': double((scheme.pulse_intervals * 1e-3).tolist()),  # Convert from ms to s.
        'tau': 1e-4,  # Time interval for waveform discretization in seconds.
    }


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

    def jacobian(self, scheme: DiffusionAcquisitionScheme) -> np.ndarray:
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
        return np.concatenate([s0 * np.array(jac), signal], axis=1)[:, self.include_optimize]

    # TODO: Implement fit method

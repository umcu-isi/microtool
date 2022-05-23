import warnings
from dataclasses import dataclass
from os import PathLike
from typing import Union, List, Tuple, Dict, Optional

import numpy as np
from pathlib import Path
from scipy.optimize import LinearConstraint


# TODO: Linear constraints? For example: Δ > δ. Deal with fixed parameters when applying constraints
@dataclass
class AcquisitionParameters:
    # noinspection PyUnresolvedReferences
    """
    Defines a series of N MR acquisition parameter values, such as a series of b-values.

    :param values: A numpy array with N parameter values.
    :param unit: The parameter unit as a string, e.g. 's/mm²'.s
    :param scale: The typical parameter value scale (order of magnitude).
    :param lower_bound: Lower constraint. None is used to specify no bound. Default: 0.
    :param upper_bound: Upper constraint. None is used to specify no bound. Default: None.
    :param fixed: Boolean indicating if the parameter is considered fixed or not (default: false).
    """
    values: np.ndarray
    unit: str
    scale: float
    lower_bound: Optional[float] = 0.0
    upper_bound: Optional[float] = None
    fixed: bool = False

    def __str__(self):
        fixed = ' (fixed parameter)' if self.fixed else ''
        return f'{self.values} {self.unit}{fixed}'


# TODO: Add function to check if all required tissue parameters are present.
class AcquisitionScheme(Dict[str, AcquisitionParameters]):
    """
    Base-class for MR acquisition schemes.

    :param parameters: A dictionary with AcquisitionParameters. Try to stick to BIDS nomenclature for the parameter
     keys.
    :raise ValueError: Lists have unequal length.
    """

    def __init__(self, parameters: Dict[str, AcquisitionParameters]):
        super().__init__()

        # Copy the acquisition parameter values into one matrix. This will raise a ValueError in case the value
        # arrays/lists are inhomogeneous.
        self._matrix = np.array([val.values for val in parameters.values()], dtype=np.float64)

        # Create a new dictionary with parameter values pointing to the _parameter_matrix.
        self.update({
            key: AcquisitionParameters(
                values=self._matrix[i],
                unit=parameters[key].unit,
                scale=parameters[key].scale,
                lower_bound=parameters[key].lower_bound,
                upper_bound=parameters[key].upper_bound,
                fixed=parameters[key].fixed
            ) for i, key in enumerate(parameters.keys())
        })

    def __str__(self) -> str:
        m, n = self._matrix.shape
        parameters = '\n'.join(f'    - {key}: {value}' for key, value in self.items())
        return f'Acquisition scheme with {n} measurements and {m} scalar parameters:\n{parameters}'

    def get_free_parameters(self) -> np.ndarray:
        """
        Returns the free acquisition parameters as an M×N matrix, where M is the number of parameters and N is the
        number of measurements in the acquisition scheme.

        :return: An M×N matrix with acquisition parameters.
        """
        mask = np.array([not p.fixed for p in self.values()])
        return self._matrix[mask].ravel()

    def set_free_parameters(self, parameters: np.ndarray) -> None:
        """
        Sets the free acquisition parameters from an M×N matrix, where M is the number of parameters and N is the
        number of measurements in the acquisition scheme. The M×N matrix may be flattened.

        :param parameters: An M×N matrix with acquisition parameters.
        """
        m, n = self._matrix.shape
        mask = np.array([not p.fixed for p in self.values()])
        mask = np.broadcast_to(mask, [n, m]).T
        np.place(self._matrix, mask, parameters)

    def get_free_parameter_scales(self) -> np.ndarray:
        """
        Returns the bounds on the free tissue parameters.

        :return: A list of M×N (min, max) pairs, where M is the number of parameters and N is the
         number of measurements in the acquisition scheme. None is used to specify no bound.
        """
        m, n = self._matrix.shape
        return np.array([p.scale for p in self.values() if not p.fixed for _ in range(n)])

    def get_free_parameter_bounds(self) -> List[Tuple[Optional[float], Optional[float]]]:
        """
        Returns the bounds on the free tissue parameters.

        :return: A list of M×N (min, max) pairs, where M is the number of parameters and N is the
         number of measurements in the acquisition scheme. None is used to specify no bound.
        """
        m, n = self._matrix.shape
        return [(p.lower_bound, p.upper_bound) for p in self.values() if not p.fixed for _ in range(n)]
    
    def get_free_parameter_keys(self) -> List[str]:
        """ 
        Function for extracting the keys of the free parameters

        :return: list of the keys of the free parameters, in the same order as get_free_parameters
        """
        return [key for key,value in self.items() if not value.fixed]

    def get_constraints(self) -> LinearConstraint:
        """
        Returns optimisation constraints on the scheme parameters.

        :return: A scipy.optimize.LinearConstraint object. None is used to specify no constraints.
        """
        # TODO: deal with fixed variables! 
        return None

class DiffusionAcquisitionScheme(AcquisitionScheme):
    """
    Defines a diffusion MR acquisition scheme.

    :param b_values: A list or numpy array of b-values in s/mm².
    :param b_vectors: A list or numpy array of direction cosines.
    :param pulse_widths: A list or numpy array of pulse widths δ in milliseconds.
    :param pulse_intervals: A list or numpy array of pulse intervals Δ in milliseconds.
    :raise ValueError: b-vectors are not unit vectors or lists have unequal length.
    """
    def __init__(self,
                 b_values: Union[List[float], np.ndarray],
                 b_vectors: Union[List[Tuple[float, float, float]], np.ndarray],
                 pulse_widths: Union[List[float], np.ndarray],
                 pulse_intervals: Union[List[float], np.ndarray]):
        
        # TODO: Check on initial pulse variables
        
        # Check if the b-vectors are unit vectors and set b=0 'vectors' to (0, 0, 0) as per convention.
        b0 = b_values == 0
        b_vectors = np.asarray(b_vectors, dtype=np.float64)
        if not np.allclose(np.linalg.norm(b_vectors[~b0], axis=1), 1):
            raise ValueError('b-vectors are not unit vectors.')
        b_vectors[b0] = 0

        # Calculate the spherical angles φ and θ.
        phi = np.arctan2(b_vectors[:, 1], b_vectors[:, 0])
        theta = np.arccos(b_vectors[:, 2])

        

        super().__init__({
            'DiffusionBValue': AcquisitionParameters(values=b_values, unit='s/mm²', scale=1000),
            'DiffusionGradientAnglePhi': AcquisitionParameters(
                values=phi, unit='rad', scale=1, lower_bound=None, fixed=True),
            'DiffusionGradientAngleTheta': AcquisitionParameters(
                values=theta, unit='rad', scale=1, lower_bound=None, fixed=True),
            'DiffusionPulseWidth': AcquisitionParameters(values=pulse_widths, unit='ms', scale=10, fixed=True),
            'DiffusionPulseInterval': AcquisitionParameters(values=pulse_intervals, unit='ms', scale=10, fixed=True),
        })

    @property
    def b_values(self) -> np.ndarray:
        """
        An array of N b-values in s/mm².
        """
        return self['DiffusionBValue'].values

    @property
    def phi(self) -> np.ndarray:
        """
        An array of N angles in radians.
        """
        return self['DiffusionGradientAnglePhi'].values

    @property
    def theta(self) -> np.ndarray:
        """
        An array of N angles in radians.
        """
        return self['DiffusionGradientAngleTheta'].values

    @property
    def pulse_widths(self) -> np.ndarray:
        """
        An array of N pulse widths in milliseconds.
        """
        return self['DiffusionPulseWidth'].values

    @property
    def pulse_intervals(self) -> np.ndarray:
        """
        An array of N pulse intervals in milliseconds.
        """
        return self['DiffusionPulseInterval'].values

    # TODO: verify results.
    @property
    def pulse_magnitude(self) -> np.ndarray:
        """
        :return: An array of N gradient magnitudes in mT/m. Assumes b = γ² G² δ² (Δ - δ/3).
        """
        b_values = self.b_values * 1e6  # Convert from s/mm² to s/m².
        pulse_widths = self.pulse_widths * 1e-3  # Convert from ms to s.
        pulse_intervals = self.pulse_intervals * 1e-3  # Convert from ms to s.
        gyromagnetic_ratio = 2.6752218744e8 * 1e-3  # Convert from 1/s/T to 1/s/mT.

        return np.sqrt(
            3 * b_values /
            (np.square(gyromagnetic_ratio * pulse_widths) * (3 * pulse_intervals - pulse_widths))
        )

    @property
    def b_vectors(self) -> np.ndarray:
        """
        An N×3 array of direction cosines.
        """
        sin_theta = np.sin(self.theta)
        b_vectors = np.array([sin_theta * np.cos(self.phi), sin_theta * np.sin(self.phi), np.cos(self.theta)]).T

        # Set b=0 'vectors' to (0, 0, 0) as per convention.
        b_vectors[self.b_values == 0] = 0

        return b_vectors

    def write_bval(self, file: Union[str, bytes, PathLike]) -> None:
        """
        Writes b-values to an FSL [*_]dwi.bval file.
        """
        # An ASCII text file containing a list of b values applied during each volume acquisition. The b values are
        # assumed to be in s/mm² units. The order of entries in this file must match the order of volumes in the input
        # data and entries in the gradient directions text file. The format is: b_1 b_2 b_3 ... b_n
        file = Path(file)
        if not file.name.endswith('_dwi.bval') and not file.name == 'dwi.bval':
            warnings.warn('BIDS specifies that FSL b-value files should be named like: [*_]dwi.bval')

        with file.open('w', encoding='latin-1', newline='\n') as f:
            f.write(' '.join(f'{x:.6e}' for x in self.b_values))

    def write_bvec(self, file) -> None:
        """
        Writes b-vectors to an FSL [*_]dwi.bvec file.
        """
        # The [*_]dwi.bvec file contains 3 rows with N space-delimited floating-point numbers (corresponding to the N
        # volumes in the corresponding NIfTI file.) The first row contains the x elements, the second row contains
        # the y elements and the third row contains the z elements of a unit vector in the direction of the applied
        # diffusion gradient, where the i-th elements in each row correspond together to the i-th volume, with [0,0,0]
        # for non-diffusion-weighted (also called b=0 or low-b) volumes. Following the FSL format for the
        # [*_]dwi.bvec specification, the coordinate system of the b vectors MUST be defined with respect to the
        # coordinate system defined by the header of the corresponding _dwi NIfTI file and not the scanner's device
        # coordinate system.
        file = Path(file)
        if not file.name.endswith('_dwi.bvec') and not file.name == 'dwi.bvec':
            warnings.warn('BIDS specifies that FSL b-vector files should be named like: [*_]dwi.bvec')

        with file.open('w', encoding='latin-1', newline='\n') as f:
            for bvec in self.b_vectors:
                f.write(' '.join(f'{x:.6e}' for x in bvec) + '\n')

    def get_constraints(self) -> LinearConstraint:
        # Matrix defining Δ > δ or equivalently 0 < Δ - δ
        pulse_num = len(self.pulse_widths)
        
        
        widthBlock = np.diag(np.repeat(-1,pulse_num))
        intervalBlock = np.diag(np.repeat(1,pulse_num))
        
        
        lb = np.repeat(0,pulse_num)
        ub = np.repeat(np.inf,pulse_num)
        return None

class InversionRecoveryAcquisitionScheme(AcquisitionScheme):
    """
    Defines an inversion-recovery MR acquisition scheme.

    :param repetition_times: A list or numpy array of repetition times TR in milliseconds.
    :param echo_times: A list or numpy array of echo times TE in milliseconds.
    :param inversion_times: A list or numpy array of inversion times TI in milliseconds.
    :raise ValueError: Lists have unequal length.
    """
    def __init__(self,
                 repetition_times: Union[List[float], np.ndarray],
                 echo_times: Union[List[float], np.ndarray],
                 inversion_times: Union[List[float], np.ndarray]):
        super().__init__(
            # TODO: Check that initial 0 < -TI + TR - TE is satisfied
            {
            'InversionTime': AcquisitionParameters(values=inversion_times, unit='ms', scale=100,lower_bound = 0.5, upper_bound=100),
            'RepetitionTimeExcitation': AcquisitionParameters(values=repetition_times, unit='ms', scale=100,lower_bound = 0.5, upper_bound=100),
            'EchoTime': AcquisitionParameters(values=echo_times, unit='ms', scale=10,lower_bound = 0.5, upper_bound=100),
        })

    @property
    def repetition_times(self) -> np.ndarray:
        """
        An array of N repetition times in milliseconds.
        """
        return self['RepetitionTimeExcitation'].values

    @property
    def echo_times(self) -> np.ndarray:
        """
        An array of N echo times in milliseconds.
        """
        return self['EchoTime'].values

    @property
    def inversion_times(self) -> np.ndarray:
        """
        An array of N inversion times in milliseconds.
        """
        return self['InversionTime'].values

    def get_constraints(self) -> LinearConstraint:
        
        pulse_num = len(self.repetition_times)
        
        # Matrix defining TR > TI + TE or equivalently 0 < -TI + TR - TE for every pulse
        TIBlock = np.diag(np.repeat(-1,pulse_num))
        TRBlock = np.diag(np.repeat(+1,pulse_num))
        TEBlock = TIBlock

        A = np.concatenate((TIBlock,TRBlock,TEBlock),axis=1)
        lb = np.repeat(0,pulse_num)
        ub = np.repeat(np.inf,pulse_num)
        return LinearConstraint(A,lb,ub)
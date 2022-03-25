import warnings
from dataclasses import dataclass
from os import PathLike
from typing import Union, List, Tuple, Dict, Optional

import numpy as np
from pathlib import Path

from microtool.tissue_model import TissueModel


# TODO: Linear constraints? For example: Δ > δ  and TR > TI + TE
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
        values = ', '.join(str(x) for x in self.values)
        fixed = ' (fixed parameter)' if self.fixed else ''
        return f'{values} {self.unit}{fixed}'


# TODO: Add function to check if all required tissue parameters are present.
class AcquisitionScheme:
    """
    Base-class for MR acquisition schemes.

    :param parameters: A dictionary with AcquisitionParameters. Try to stick to BIDS nomenclature for the parameter
     keys.
    :raise ValueError: Lists have unequal length.
    """
    _parameters: Dict[str, AcquisitionParameters]
    _parameter_matrix: np.ndarray

    def __init__(self, parameters: Dict[str, AcquisitionParameters]):
        # Copy the acquisition parameter values into one matrix. This will raise a ValueError in case the value
        # arrays/lists are inhomogeneous.
        self._parameter_matrix = np.array([val.values for val in parameters.values()], dtype=np.float64)

        # Create a new dictionary with parameter values pointing to the _parameter_matrix.
        self._parameters = {
            key: AcquisitionParameters(
                values=self._parameter_matrix[i],
                unit=parameters[key].unit,
                scale=parameters[key].scale,
                lower_bound=parameters[key].lower_bound,
                upper_bound=parameters[key].upper_bound,
                fixed=parameters[key].fixed
            ) for i, key in enumerate(parameters.keys())
        }

    def __str__(self) -> str:
        m, n = self._parameter_matrix.shape
        parameters = '\n'.join(f'    {key}: {value}' for key, value in self._parameters.items())
        return f'Acquisition scheme with {n} measurements and {m} scalar parameters:\n{parameters}'

    def __call__(self, model: TissueModel) -> np.ndarray:
        """
        Calculates the signal attenuation for all measurements in this acquisition scheme as a function of the tissue
        model.

        :param model: a TissueModel defining the MR tissue properties.
        :return: An array with signal attenuation values.
        """
        raise NotImplementedError

    def model_jacobian(self, model: TissueModel) -> np.ndarray:
        """
        Calculates the change in signal attenuation for all measurements in this acquisition scheme due to a change in
        the tissue model parameters.

        :param model: a TissueModel defining the MR tissue properties.
        :return: An N×M Jacobian matrix, where N is the number of samples and M is the number of parameters.
        """
        raise NotImplementedError

    def get_model_scales(self, model: TissueModel) -> np.ndarray:
        """
        Returns the parameter scales of the relevant tissue parameters.

        :param model: a TissueModel defining the MR tissue properties.
        :return: An array with M parameter scales.
        """
        raise NotImplementedError

    def get_free_parameters(self):
        """
        Returns the free acquisition parameters as an M×N matrix, where M is the number of parameters and N is the
        number of measurements in the acquisition scheme.

        :return: An M×N matrix with acquisition parameters.
        """
        mask = np.array([not p.fixed for p in self._parameters.values()])
        return self._parameter_matrix[mask].ravel()

    def set_free_parameters(self, parameters: np.ndarray) -> None:
        """
        Sets the free acquisition parameters from an M×N matrix, where M is the number of parameters and N is the
        number of measurements in the acquisition scheme. The M×N matrix may be flattened.

        :param parameters: An M×N matrix with acquisition parameters.
        """
        m, n = self._parameter_matrix.shape
        mask = np.array([not p.fixed for p in self._parameters.values()])
        mask = np.broadcast_to(mask, [n, m]).T
        np.place(self._parameter_matrix, mask, parameters)

    def get_free_parameter_scales(self) -> np.ndarray:
        """
        Returns the bounds on the free tissue parameters.

        :return: A list of M×N (min, max) pairs, where M is the number of parameters and N is the
         number of measurements in the acquisition scheme. . None is used to specify no bound.
        """
        m, n = self._parameter_matrix.shape
        return np.array([p.scale for p in self._parameters.values() if not p.fixed for _ in range(n)])

    def get_free_parameter_bounds(self) -> List[Tuple[Optional[float], Optional[float]]]:
        """
        Returns the bounds on the free tissue parameters.

        :return: A list of M×N (min, max) pairs, where M is the number of parameters and N is the
         number of measurements in the acquisition scheme. . None is used to specify no bound.
        """
        m, n = self._parameter_matrix.shape
        return [(p.lower_bound, p.upper_bound) for p in self._parameters.values() if not p.fixed for _ in range(n)]


class DiffusionAcquisitionScheme(AcquisitionScheme):
    """
    Defines a diffusion MR acquisition scheme.

    :param b_values: A list or numpy array of b-values in s/mm².
    :param b_vectors: A list or numpy array of direction cosines.
    :param pulse_widths: A list or numpy array of pulse widths δ in seconds.
    :param pulse_intervals: A list or numpy array of pulse intervals Δ in seconds.
    :raise ValueError: b-vectors are not unit vectors or lists have unequal length.
    """
    def __init__(self,
                 b_values: Union[List[float], np.ndarray],
                 b_vectors: Union[List[Tuple[float, float, float]], np.ndarray],
                 pulse_widths: Union[List[float], np.ndarray],
                 pulse_intervals: Union[List[float], np.ndarray]):
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
            'DiffusionBValue': AcquisitionParameters(values=b_values, unit='s/mm²', scale=1e3),
            'DiffusionGradientAnglePhi': AcquisitionParameters(
                values=phi, unit='rad', scale=1, lower_bound=None, fixed=True),
            'DiffusionGradientAngleTheta': AcquisitionParameters(
                values=theta, unit='rad', scale=1, lower_bound=None, fixed=True),
            'DiffusionPulseWidth': AcquisitionParameters(values=pulse_widths, unit='s', scale=1, fixed=True),
            'DiffusionPulseInterval': AcquisitionParameters(values=pulse_intervals, unit='s', scale=1, fixed=True),
        })

    def get_b_values(self) -> np.ndarray:
        """
        Returns the pulse b-values.

        :return: An array of N b-values in s/mm².
        """
        return self._parameters['DiffusionBValue'].values

    def get_phi(self) -> np.ndarray:
        """
        Returns the pulse gradient direction angles φ.

        :return: An array of N angles in radians.
        """
        return self._parameters['DiffusionGradientAnglePhi'].values

    def get_theta(self) -> np.ndarray:
        """
        Returns the pulse gradient direction angles θ.

        :return: An array of N angles in radians.
        """
        return self._parameters['DiffusionGradientAngleTheta'].values

    def get_pulse_widths(self) -> np.ndarray:
        """
        Returns the pulse durations δ.

        :return: An array of N pulse widths in seconds.
        """
        return self._parameters['DiffusionPulseWidth'].values

    def get_pulse_intervals(self) -> np.ndarray:
        """
        Returns the pulse intervals Δ.

        :return: An array of N pulse intervals in seconds.
        """
        return self._parameters['DiffusionPulseInterval'].values

    # TODO: verify results.
    def get_pulse_magnitude(self) -> np.ndarray:
        """
        Returns the gradient magnitude of the pulses. Assumes b = γ² G² δ² (Δ - δ/3).

        :return: An array of N gradient magnitudes in mT/m.
        """
        b_values = self.get_b_values() * 1e3  # Convert from s/mm² to s/m².
        pulse_widths = self.get_pulse_widths()  # s
        pulse_intervals = self.get_pulse_intervals()  # s
        gyromagnetic_ratio = 2.6752218744e8 * 1e-3  # Convert from 1/s/T to 1/s/mT.

        return np.sqrt(
            3 * b_values /
            (np.square(gyromagnetic_ratio * pulse_widths) * (3 * pulse_intervals - pulse_widths))
        )

    def get_b_vectors(self) -> np.ndarray:
        """
        Calculates the b-vectors.

        :return: An N×3 array of direction cosines.
        """
        phi = self.get_phi()
        theta = self.get_theta()
        sin_theta = np.sin(theta)
        b_vectors = np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), np.cos(theta)]).T

        # Set b=0 'vectors' to (0, 0, 0) as per convention.
        b_values = self.get_b_values()
        b_vectors[b_values == 0] = 0

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
            bval = self.get_b_values()
            f.write(' '.join(f'{x:.6e}' for x in bval))

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
            for bvec in self.get_b_vectors():
                f.write(' '.join(f'{x:.6e}' for x in bvec) + '\n')

    def __call__(self, model: TissueModel) -> np.ndarray:
        b_values = self.get_b_values()
        b_vectors = self.get_b_vectors()
        pulse_widths = self.get_pulse_widths()
        pulse_intervals = self.get_pulse_intervals()
        return model.s0 * model.diffusion_model(b_values, b_vectors, pulse_widths, pulse_intervals)

    def model_jacobian(self, model: TissueModel) -> np.ndarray:
        b_values = self.get_b_values()
        b_vectors = self.get_b_vectors()
        pulse_widths = self.get_pulse_widths()
        pulse_intervals = self.get_pulse_intervals()

        s0_jac = model.diffusion_model(b_values, b_vectors, pulse_widths, pulse_intervals).reshape((-1, 1))
        model_jac = model.diffusion_model.jacobian(b_values, b_vectors, pulse_widths, pulse_intervals)
        return np.concatenate((s0_jac, model.s0 * model_jac), axis=1)

    def get_model_scales(self, model: TissueModel) -> np.ndarray:
        # Use the S0 parameter value as scale.
        return np.concatenate(([model.s0], model.diffusion_model.get_scales()))


class InversionRecoveryAcquisitionScheme(AcquisitionScheme):
    """
    Defines an inversion-recovery MR acquisition scheme. Rather than varying TR to achieve different T1 weightings,
    Mulkern et al. (2000) incorporate an inversion pulse prior to the 90° pulse in the diffusion-weighted SE sequence
    for simultaneous D-T1 measurement.

    See section 7.4.2 of 'Advanced Diffusion Encoding Methods in MRI', Topgaard D, editor (2020):
    https://www.ncbi.nlm.nih.gov/books/NBK567564

    :param repetition_times: A list or numpy array of repetition times TR in seconds.
    :param echo_times: A list or numpy array of echo times TE in seconds.
    :param inversion_times: A list or numpy array of inversion times TI in seconds.
    :raise ValueError: Lists have unequal length.
    """
    def __init__(self,
                 repetition_times: Union[List[float], np.ndarray],
                 echo_times: Union[List[float], np.ndarray],
                 inversion_times: Union[List[float], np.ndarray]):
        super().__init__({
            'InversionTime': AcquisitionParameters(values=inversion_times, unit='s', scale=1),
            'RepetitionTimeExcitation': AcquisitionParameters(values=repetition_times, unit='s', scale=1),
            'EchoTime': AcquisitionParameters(values=echo_times, unit='s', scale=1),
        })

    def get_repetition_times(self) -> np.ndarray:
        """
        Returns repetition times.

        :return: An array of N repetition times in seconds.
        """
        return self._parameters['RepetitionTimeExcitation'].values

    def get_echo_times(self) -> np.ndarray:
        """
        Returns the echo times.

        :return: An array of N echo times in seconds.
        """
        return self._parameters['EchoTime'].values

    def get_inversion_times(self) -> np.ndarray:
        """
        Returns the inversion times.

        :return: An array of N inversion times in seconds.
        """
        return self._parameters['InversionTime'].values

    def __call__(self, model: TissueModel) -> np.ndarray:
        ti = self._parameters['InversionTime'].values
        tr = self._parameters['RepetitionTimeExcitation'].values
        te = self._parameters['EchoTime'].values

        ti_t1 = np.exp(-ti / model.t1)
        tr_t1 = np.exp(-tr / model.t1)
        te_t2 = np.exp(-te / model.t2)

        return model.s0 * (1 - 2 * ti_t1 + tr_t1) * te_t2

    def model_jacobian(self, model: TissueModel) -> np.ndarray:
        ti = self._parameters['InversionTime'].values
        tr = self._parameters['RepetitionTimeExcitation'].values
        te = self._parameters['EchoTime'].values

        ti_t1 = np.exp(-ti / model.t1)
        tr_t1 = np.exp(-tr / model.t1)
        te_t2 = np.exp(-te / model.t2)

        # Calculate the derivative of the signal attenuation to S0, T1 and to T2.
        return np.array([
            (1 - 2 * ti_t1 + tr_t1) * te_t2,   # δS(S0, T1, T2) / δS0
            model.s0 * (-2 * ti * ti_t1 + tr * tr_t1) * te_t2 / (model.t1 ** 2),  # δS(S0, T1, T2) / δT1
            model.s0 * te * (1 - 2 * ti_t1 + tr_t1) * te_t2 / (model.t2 ** 2)  # δS(S0, T1, T2) / δT2
        ]).T

    def get_model_scales(self, model: TissueModel) -> np.ndarray:
        # Use the S0, T1 and T2 parameter values as scale.
        return np.array([model.s0, model.t1, model.t2])

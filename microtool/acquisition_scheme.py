import warnings
from abc import ABC, abstractmethod
from copy import copy
from math import prod
from os import PathLike
from pathlib import Path
from typing import Union, List, Tuple, Dict, Optional

import numpy as np
from scipy.optimize import NonlinearConstraint, LinearConstraint
from tabulate import tabulate

from microtool.utils.solve_echo_time import minimal_echo_time

ConstraintTypes = Union[
    NonlinearConstraint, LinearConstraint, List[Union[LinearConstraint, NonlinearConstraint]]]


class AcquisitionParameters:
    """
    Defines a series of N MR acquisition parameter values, such as a series of b-values.

    :param values: A numpy array with N parameter values.
    :param unit: The parameter unit as a string, e.g. 's/mm²'.s
    :param scale: The typical parameter value scale (order of magnitude).
    :param symbol: A string used in type setting
    :param lower_bound: Lower constraint. None is used to specify no bound. Default: 0.
    :param upper_bound: Upper constraint. None is used to specify no bound. Default: None.
    :param fixed: Boolean indicating if the parameter is considered fixed or not (default: false).
    """

    def __init__(self, values: np.ndarray,
                 unit: str,
                 scale: float,
                 symbol: Optional[str] = None,
                 lower_bound: Optional[float] = 0.0,
                 upper_bound: Optional[float] = None,
                 fixed: bool = False):

        self._check_bounded(values, lower_bound, upper_bound)
        self.values = values
        self.unit = unit
        self.scale = scale
        self.symbol = symbol
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.fixed = fixed
        self._optimize_mask = np.tile(not self.fixed, self.values.shape)

    def __str__(self):
        fixed = ' (fixed parameter)' if self.fixed else ''
        return f'{self.values} {self.unit}{fixed}'

    def __len__(self):
        return len(self.values)

    def set_fixed_mask(self, fixed_mask: np.ndarray) -> None:
        """
        Set the measurements you wish to stay fixed during optimization of this parameter.

        :param fixed_mask: The boolean np.array that evaluates True for values that need to stay fixed
        """
        # check the shapes
        if fixed_mask.shape != self.values.shape:
            raise ValueError(f"Value_mask shape {fixed_mask.shape} does not match value shape {self.values.shape}")

        # update parameter fixed if all measurements are fixed
        self.fixed = np.all(fixed_mask)
        # the optimization mask is the negation of the fixed mask
        self._optimize_mask = np.logical_not(fixed_mask)

    def set_free_values(self, new_values: np.ndarray) -> None:
        """
        Change the value of only the parameter values that should be optimised

        :param new_values: The values to be assigned
        """
        self.values[self._optimize_mask] = new_values

    @property
    def free_values(self) -> np.ndarray:
        return self.values[self._optimize_mask]

    @property
    def optimize_mask(self):
        return self._optimize_mask

    @staticmethod
    def _check_bounded(values, lower_bound, upper_bound):
        """
        :raises ValueError: If the parameter values are out of bounds
        """
        flag = False
        if lower_bound:
            if np.any((values < lower_bound)):
                flag = True
        if upper_bound:
            if np.any(values > upper_bound):
                flag = True
        if flag:
            raise ValueError("One or more parameter values are out of bounds.")


class AcquisitionScheme(Dict[str, AcquisitionParameters], ABC):
    """
    Base-class for MR acquisition schemes.

    :param parameters: A dictionary with AcquisitionParameters. Try to stick to BIDS nomenclature for the parameter
     keys.
    :raise ValueError: Lists have unequal length.
    """

    def __init__(self, parameters: Dict[str, AcquisitionParameters]):
        # Check that all parameters have the same length
        self._check_parameter_lengths(parameters)
        super().__init__(parameters)

    def __str__(self) -> str:

        table = {}
        for key, value in self.items():
            if value.fixed:
                entry = {f"{key} [{value.unit}] (fixed)": value.values}
            else:
                entry = {f"{key} [{value.unit}] in {value.lower_bound, value.upper_bound}": value.values}

            table.update(entry)

        table_str = tabulate(table, headers='keys')
        return f'Acquisition scheme with {self.pulse_count} measurements and {len(self)} scalar parameters:\n{table_str}'

    @staticmethod
    def _check_parameter_lengths(parameters: Dict[str, AcquisitionParameters]):
        # making a dict with parameter lengths for more informative error message
        parameter_lengths = {}
        for key, parameter in parameters.items():
            parameter_lengths[key] = len(parameter)

        # Checking against the first parameters length
        all_lenghts = list(parameter_lengths.values())
        first = all_lenghts[0]
        for length in all_lenghts:
            if length != first:
                raise ValueError(f"One or more parameters have unequal length: {parameter_lengths}")

    @property
    def free_parameters(self) -> Dict[str, np.ndarray]:
        """
        Returns the free acquisition parameters as a dictionary of AcquisitionParameter name : values

        :return: A dictionary containing key : TissueParameter.values pairs.
        """
        return {key: self[key].free_values for key in self.free_parameter_keys}

    @property
    def free_parameter_vector(self) -> np.ndarray:
        """
        Returns all the free parameters as a flattened vector.
        :return:
        """
        return np.concatenate([val.free_values.flatten() for val in self.values() if not val.fixed])

    def set_free_parameter_vector(self, vector: np.ndarray) -> None:
        """
        A setter function for the free parameters. Infers the desired parameter by shape information
        :param vector: The parameter vector you wish to assign to the scheme
        :return: None, changes parameter attributes
        """
        # Reshape the flattened vector based on parameter value shapes
        i = 0
        for key in self.free_parameter_keys:
            # shape of the current parameter free values determines the number of values we can assign.
            shape = self[key].free_values.shape
            # computing how many values of the vector belong to the current parameter
            stride = int(prod(shape))
            new_values = vector[i:(i + stride)]
            self[key].set_free_values(new_values.reshape(shape))
            i += stride

    def get_free_parameter_idx(self, parameter: str, pulse_id: int) -> int:
        """
        Allows you to get the index in the free parameter vector of a parameter pulse number combination
        :param parameter: The name of free acquisition parameter
        :param pulse_id: The pulse for which you need the index
        :return: The index
        """
        i = 0
        for key in self.free_parameter_keys:
            if key == parameter:
                return i + pulse_id
            shape = self[key].free_values.shape
            stride = int(prod(shape))
            i += stride

    def set_free_parameter_bounds(self, bounds: List[Tuple[float, float]]) -> None:
        """
        Setter function for the parameter bounds, requires you to provide bounds for all parameters in sequence
        :param bounds: The bounds
        :return: None, changes parameter attributes
        """
        if len(bounds) != len(self.free_parameter_keys):
            raise ValueError("provide bounds only for free parameters.")
        for i, key in enumerate(self.free_parameter_keys):
            self[key].lower_bound = bounds[i][0]
            self[key].upper_bound = bounds[i][1]

    @property
    def free_parameter_scales(self) -> np.ndarray:
        """
        Getter for the scale of the free parameters repeated for the number of pulses.

        :return: The scale of the free parameters
        """
        return np.repeat(np.array([p.scale for p in self.values() if not p.fixed]),
                         [len(p.free_values) for p in self.values() if not p.fixed])

    @property
    def free_parameter_bounds_scaled(self) -> List[Tuple[Optional[float], ...]]:
        """
        :return: The free parameters bounds divided by their scales. List of min max pairs
        """

        bounds = []
        for key in self.free_parameter_keys:
            p = self[key]
            p_bounds = (p.lower_bound, p.upper_bound)
            for _ in range(len(p.free_values)):
                bounds.append(tuple([None if bound is None else bound / p.scale for bound in p_bounds]))

        return bounds

    @property
    def free_parameter_keys(self) -> List[str]:
        """
        Function for extracting the keys of the free parameters

        :return: list of the keys of the free parameters, in the same order as free_parameters
        """
        return [key for key, value in self.items() if not value.fixed]

    @property
    def pulse_count(self) -> int:
        max_parameter_length = 0
        for parameter in self.free_parameters:
            par_length = len(self.free_parameters[parameter])
            if par_length > max_parameter_length:
                max_parameter_length = par_length

        return max_parameter_length

    def get_parameter_from_parameter_vector(self, parameter: str, x: np.ndarray) -> np.ndarray:
        """
        This function helps to get a free parameter from a scipy array. Usefull for building constraints on specific
        parameters. Be warned the function does not check if the parameter you require is actually a free parameter!!

        :param parameter: The name of the free parameter you wish to extract from the scipy array.
        :param x: The scipy array of free parameters, in scaled units.
        :return: The rescaled parameter values from the scipy array (so now in physical units).
        """
        i = 0
        for key in self.free_parameter_keys:
            scale = self[key].scale
            shape = self[key].free_values.shape
            stride = int(prod(shape))
            if parameter == key:
                return x[i:(i + stride)].reshape(shape) * scale
            i += stride

    @abstractmethod
    def get_constraints(self) -> ConstraintTypes:
        """
        Returns optimisation constraints on the scheme parameters. Implementation is child-class specific.

        :return: A scipy.optimize.LinearConstraint object. None is used to specify no constraints.
        The constraint is defined by lb <= A.x <= ub, x being the array of parameters optimized.
        A is the matrix defining the constraint relation between parameters.
        """
        raise NotImplementedError()

    def _copy_and_update_parameter(self, parameter: str, x: np.ndarray):
        """
        Makes a copy from Acquisition parameter values and inserts the values suggested by the optimizer in
        x.
        :param parameter: The name of the parameter for which you want the update full value array
        :param x: The optimizer suggestion for ALL free parameters
        :return: The copy of all values and the updated free parameter values
        """
        update_values = self.get_parameter_from_parameter_vector(parameter, x)
        value_array = copy(self[parameter].values)
        value_array[self[parameter].optimize_mask] = update_values
        return value_array

    def _are_fixed(self, parameter_list: List[str]) -> bool:
        are_fixed = [self[name].fixed for name in parameter_list]
        return all(are_fixed)


class DiffusionAcquisitionScheme(AcquisitionScheme):
    """
    Defines a diffusion MR acquisition scheme. (PGSE?)

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

        # Checking the constraint on delta and Delta
        if np.any(pulse_widths > pulse_intervals):
            raise ValueError(
                "Invalid DiffusionAcquisitionScheme: atleast one measurement with pulse_width > pulse_interval")

        # Check if the b-vectors are unit vectors and set b=0 'vectors' to (0, 0, 0) as per convention.
        b0 = b_values == 0
        if not np.any(b0):
            raise ValueError("No b0 measurements detected. The b0 measurements are required to estimate S0.")

        b_vectors = np.asarray(b_vectors, dtype=np.float64)
        if not np.allclose(np.linalg.norm(b_vectors[~b0], axis=1), 1):
            raise ValueError('b-vectors are not unit vectors.')
        b_vectors[b0] = 0

        # Calculate the spherical angles φ and θ.
        phi = np.arctan2(b_vectors[:, 1], b_vectors[:, 0])
        theta = np.arccos(b_vectors[:, 2])

        super().__init__({
            'DiffusionBValue': AcquisitionParameters(
                values=b_values, unit='s/mm²', scale=1e3, symbol="b", lower_bound=0.0, upper_bound=20e3
            ),
            'DiffusionGradientAnglePhi': AcquisitionParameters(
                values=phi, unit='rad', scale=1., symbol=r"$\phi$", lower_bound=None, fixed=True
            ),
            'DiffusionGradientAngleTheta': AcquisitionParameters(
                values=theta, unit='rad', scale=1., symbol=r"$\theta$", lower_bound=None, fixed=True
            ),
            'DiffusionPulseWidth': AcquisitionParameters(
                values=pulse_widths, unit='ms', scale=10., symbol=r"$\delta$", fixed=False, lower_bound=1.,
                upper_bound=1e2
            ),
            'DiffusionPulseInterval': AcquisitionParameters(
                values=pulse_intervals, unit='ms', scale=10., symbol=r"$\Delta$", fixed=False, lower_bound=1.,
                upper_bound=1e3
            ),
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

    def get_constraints(self) -> Optional[NonlinearConstraint]:
        # Defining Δ > δ or equivalently 0 < Δ - δ < \infty

        # We check in case both parameters are fixed we need not apply a constraint
        if self._are_fixed(['DiffusionPulseWidth', 'DiffusionPulseInterval']):
            return None

        # get non free parameter values and mask with the free parameter mask
        def delta_constraint_fun(x: np.ndarray):
            """ Should be larger than zero """
            delta = self._copy_and_update_parameter('DiffusionPulseWidth', x)
            Delta = self._copy_and_update_parameter('DiffusionPulseInterval', x)
            return Delta - delta

        return NonlinearConstraint(delta_constraint_fun, 0.0, np.inf, keep_feasible=True)


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
                 inversion_times: Union[List[float], np.ndarray],
                 ):
        # check constraint TR > TE + TI
        if np.any(repetition_times < echo_times + inversion_times):
            raise ValueError("Invalid inversion recovery scheme: atleast one measurement breaks constrain TR>TE+TI")

        # delegate to baseclass
        super().__init__(
            {
                'RepetitionTimeExcitation': AcquisitionParameters(
                    values=repetition_times, unit='ms', scale=100, symbol=r"$T_R$", lower_bound=10.0, upper_bound=1e4),
                'EchoTime': AcquisitionParameters(
                    values=echo_times, unit='ms', scale=10, symbol=r"$T_E$", fixed=True, lower_bound=.1,
                    upper_bound=1e3),
                'InversionTime': AcquisitionParameters(
                    values=inversion_times, unit='ms', scale=100, symbol=r"$T_I$", lower_bound=10.0, upper_bound=1e4)
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

    def get_constraints(self) -> Optional[NonlinearConstraint]:
        involved_parameters = ['InversionTime', 'EchoTime', 'RepetitionTimeExcitation']
        if self._are_fixed(involved_parameters):
            return None

        def time_constraint_fun(x: np.ndarray):
            # require return value larger than zero to enforce constraint TR > TE + TI
            ti = self._copy_and_update_parameter('InversionTime', x)
            te = self._copy_and_update_parameter('EchoTime', x)
            tr = self._copy_and_update_parameter('RepetitionTimeExcitation', x)
            return tr - te - ti

        return NonlinearConstraint(time_constraint_fun, 0.0, np.inf)


class EchoScheme(AcquisitionScheme):
    def __init__(self, TE: np.ndarray):
        super().__init__({
            'EchoTime': AcquisitionParameters(values=TE, unit='ms', scale=1, symbol=r"$T_E$", lower_bound=5,
                                              upper_bound=200)
        })

    @property
    def echo_times(self):
        return self['EchoTime'].values

    def get_constraints(self) -> Optional[Union[dict, List[dict]]]:
        """ For now without constraints. """
        return None


class FlaviusAcquisitionScheme(AcquisitionScheme):
    """

    :param b_values:
    :param echo_times:
    :param max_gradient:
    """

    def __init__(self, b_values: Union[List[float], np.ndarray], echo_times: Union[List[float], np.ndarray],
                 max_gradient: np.ndarray,
                 max_slew_rate: np.ndarray,
                 half_readout_time: np.ndarray,
                 excitation_time_pi: np.ndarray,
                 excitation_time_half_pi: np.ndarray
                 ):
        # Check for b0 values? make sure initial scheme satisfies constraints.

        super().__init__({
            'DiffusionBvalue': AcquisitionParameters(
                values=b_values, unit='s/mm^2', scale=1000, symbol=r"$b$", lower_bound=0.0, upper_bound=3e4
            ),
            'EchoTime': AcquisitionParameters(
                values=echo_times, unit='ms', scale=10, symbol=r"$T_E$", lower_bound=0, upper_bound=1e3
            ),
            'MaxPulseGradient': AcquisitionParameters(
                values=max_gradient, unit='mT/mm', scale=1, symbol=r"$G_{max}$", fixed=True
            ),
            'MaxSlewRate': AcquisitionParameters(
                values=max_slew_rate, unit='mT/mm/ms', scale=1, symbol=r"$SR$", fixed=True
            ),
            'RiseTime': AcquisitionParameters(
                values=max_gradient / max_slew_rate, unit='ms', scale=1, symbol=r"$t_{rise}$", fixed=True
            ),
            'HalfReadTime': AcquisitionParameters(
                values=half_readout_time, unit='ms', scale=10, symbol=r"$t_{half}$", fixed=True
            ),
            'PulseDurationPi': AcquisitionParameters(
                values=excitation_time_pi, unit='ms', scale=10, symbol=r"$t_{\pi}$", fixed=True
            ),
            'PulseDurationHalfPi': AcquisitionParameters(
                values=excitation_time_half_pi, unit='ms', scale=10, symbol=r"$t_{\pi / 2}$", fixed=True
            )
        })

    @property
    def echo_times(self):
        return self['EchoTime'].values

    @property
    def b_values(self):
        return self['DiffusionBvalue'].values

    def get_constraints(self) -> NonlinearConstraint:
        t180 = self['PulseDurationPi'].values
        t90 = self['PulseDurationHalfPi'].values
        G_max = self['MaxPulseGradient'].values
        t_rise = self['RiseTime'].values
        t_half = self['HalfReadTime'].values

        def fun(x: np.ndarray) -> np.ndarray:
            # get b-values from x
            b = self.get_parameter_from_parameter_vector('DiffusionBvalue', x)
            # note that b is in s/mm^2 but all other time dimensions are ms.
            # so we convert to ms/mm^2
            b *= 1e3
            # get echotimes from x, (units are # ms)
            TE = self.get_parameter_from_parameter_vector('EchoTime', x)
            # compute the minimal echotimes associated with b-values and other parameters
            TE_min = minimal_echo_time(b, t90, t180, t_half, G_max, t_rise)

            # The constraint is satisfied if actual TE is higher than minimal TE
            return TE - TE_min

        return NonlinearConstraint(fun, 0.0, np.inf)

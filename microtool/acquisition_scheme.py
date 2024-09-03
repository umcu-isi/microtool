import warnings
from abc import ABC, abstractmethod
from copy import copy
from os import PathLike
from pathlib import Path
from typing import Union, List, Tuple, Dict, Optional

import numpy as np
from scipy.optimize import NonlinearConstraint
from tabulate import tabulate

from microtool.gradient_sampling.utils import unitvector_to_angles, angles_to_unitvectors
from microtool.gradient_sampling import sample_uniform_half_sphere
from microtool.scanner_parameters import ScannerParameters, default_scanner
from microtool.utils.math import is_smaller_than_with_tolerance, is_higher_than_with_tolerance
from microtool.utils.solve_echo_time import minimal_echo_time, New_minimal_echo_time
from .constants import ConstraintTypes, GAMMA, GRADIENT_UNIT, PULSE_TIMING_UNIT, PULSE_TIMING_LB, PULSE_TIMING_UB, \
    PULSE_TIMING_SCALE,  B_VAL_LB, B_VAL_UB, B_VAL_SCALE, MAX_TE, B_MAX
from .pulse_relations import get_b_value_complete, get_gradients
from .bval_delta_pulse_relations import delta_Delta_from_TE, b_val_from_delta_Delta, constrained_dependencies


class AcquisitionParameters:
    """
    Defines a series of N MR acquisition parameter values, such as a series of b-values.

    Note: Set fixed values before introducing repeated values through the set_repetition period method to include fixed
     measurements before the repeated measurements.

    :param values: A numpy array with N parameter values.
    :param unit: The parameter unit as a string, e.g. 's/mm²'.
    :param scale: The typical parameter value scale (order of magnitude).
    :param symbol: A string used in type setting
    :param lower_bound: Lower constraint. None is used to specify no bound. Default: 0.
    :param upper_bound: Upper constraint. None is used to specify no bound. Default: None.
    :param fixed: Boolean indicating if the parameter is considered fixed or not (default: false).
    """

    def __init__(self,
                 values: np.ndarray,
                 unit: str,
                 scale: float,
                 symbol: Optional[str] = None,
                 lower_bound: Optional[float] = 0.0,
                 upper_bound: Optional[float] = None,
                 fixed: bool = False):

        self.values = values.copy().ravel()  # Ensure that the values are a 1D vector.
        self.unit = unit
        self.scale = scale
        self.symbol = symbol
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self._fixed = fixed
        self._optimize_mask = np.full(self.values.size, not self._fixed)
        self._repetition_period = 0

        if ((self.lower_bound and np.any(self.values < self.lower_bound)) or
                (self.upper_bound and np.any(self.values > self.upper_bound))):
            raise ValueError("One or more parameter values are out of bounds.")

    def __str__(self):
        fixed = ' (fixed parameter)' if self.fixed else ''
        return f'{self.values} {self.unit}{fixed}'

    def __len__(self):
        return len(self.values)

    def set_fixed_mask(self, fixed_mask: np.ndarray) -> None:
        """
        Set the measurements you wish to stay fixed during optimization of this parameter.

        :param fixed_mask: A boolean np.array indicating which values need to stay fixed.
        """
        fixed_mask = fixed_mask.ravel()

        # check the shapes
        if fixed_mask.size != self.values.size:
            raise ValueError(
                f"Value_mask size {fixed_mask.size} does not match the number of parameters {self.values.size}")

        # update parameter fixed if all measurements are fixed
        self._fixed = np.all(fixed_mask)
        # the optimization mask is the negation of the fixed mask
        self._optimize_mask = np.logical_not(fixed_mask)

    @property
    def free_values(self) -> np.ndarray:
        return self.values[self._optimize_mask]

    @free_values.setter
    def free_values(self, new_values: np.ndarray) -> None:
        """
        Change the value of only the parameter values that should be optimised.

        :param new_values: The values to be assigned
        """
        self.values[self._optimize_mask] = new_values

    @property
    def optimize_mask(self) -> np.ndarray:
        return self._optimize_mask

    def set_repetition_period(self, n: int):
        """
        Breaks up the sequence of m free acquisition parameters into m/n repetitions, reducing the number of free
        parameters from m to n.

        The free values for parameters will always be stored in the first n measurements, and the remaining m-n values
        will be fixed.

        Note: To update the fixed parameters, call update_repeated_values.

        :param n: Length of the repeated sequence (period).
        """
        n_total = np.sum(self.optimize_mask)
        if n_total % n != 0:
            raise ValueError(
                f"The repetition period ({n}) does not match with the number of free parameters ({n_total})")
        self._repetition_period = n

        # Make a mask for the free parameters.
        mask = np.zeros(shape=n_total, dtype=bool)
        mask[::self._repetition_period] = True

        # Apply the mask to the free parameters.
        optimize_mask = self.optimize_mask
        optimize_mask[self.optimize_mask] = mask
        fixed_mask = np.logical_not(optimize_mask)

        self.set_fixed_mask(fixed_mask)

    def update_repeated_values(self):
        """
        Sets the parameter values from the first value after every repetition period, starting from the first free
        measurement.
        """
        if self._repetition_period == 0:
            return

        period = self._repetition_period

        # TODO: don't rely on argmax, but store the positions of the repeated parameters.
        first_free_measurement = np.argmax(self.optimize_mask)
        for i in range(first_free_measurement, len(self.values), period):
            self.values[i + 1:i + period] = self.values[i]

    @property
    def fixed(self):
        return self._fixed

    @fixed.setter
    def fixed(self, new_val: bool):
        self._fixed = new_val
        # we should also update the mask if we change the fixed property
        self._optimize_mask = np.zeros(self._optimize_mask.size, dtype=bool)


class AcquisitionScheme(Dict[str, AcquisitionParameters], ABC):
    """
    Base-class for MR acquisition schemes.

    :param parameters: A dictionary with AcquisitionParameters. Try to stick to BIDS nomenclature for the parameter
     keys.
    :raise ValueError: Sequences of AcquisitionParameters have different lengths.
    """

    def __init__(self, parameters: Dict[str, AcquisitionParameters]):
        # Check that all parameters have the same length
        self._check_parameter_lengths(parameters)
        super().__init__(parameters)

    def __str__(self) -> str:
        table = {}
        optimization_parameters = 0
        for key, value in self.items():
            if value.fixed:
                entry = {f"{key} [{value.unit}] (fixed)": value.values}
            else:
                entry = {f"{key} [{value.unit}] in {value.lower_bound, value.upper_bound}": value.values}
                optimization_parameters += np.sum(self[key].optimize_mask)

            table.update(entry)

        table_str = tabulate(table, headers='keys')
        return f'Acquisition scheme with {self.pulse_count} measurements and {len(self)} scalar parameters. \n' \
               f'total number of optimized parameters is {optimization_parameters}:\n{table_str}'

    @staticmethod
    def _check_parameter_lengths(parameters: Dict[str, AcquisitionParameters]):
        # making a dict with parameter lengths for more informative error message
        parameter_lengths = {}
        for key, parameter in parameters.items():
            parameter_lengths[key] = len(parameter)

        # Checking against the first parameters length
        all_lengths = list(parameter_lengths.values())
        first = all_lengths[0]
        if any(n != first for n in all_lengths):
            raise ValueError(f"One or more parameters have unequal length: {parameter_lengths}")

    @property
    def free_parameters(self) -> Dict[str, np.ndarray]:
        """
        Returns the free acquisition parameters as a dictionary of AcquisitionParameter name : values

        :return: a dictionary of AcquisitionParameter name : values
        """
        return {key: self[key].free_values for key in self.free_parameter_keys}

    @property
    def free_parameter_vector(self) -> np.ndarray:
        """
        Returns all the free parameters as a flattened vector.
        :return: All free parameter values
        """
        return np.concatenate([val.free_values for val in self.values()])

    def set_free_parameter_vector(self, vector: np.ndarray) -> None:
        """
        A setter function for the free parameters.

        :param vector: The parameter vector you wish to assign to the scheme
        """
        vector = vector.ravel()
        if self.free_parameter_vector.size != vector.size:
            raise ValueError(
                "New free parameter vector does not contain the same number of values as there are free parameters.")

        # Revert the concatenation done in free_parameter_vector().
        i = 0
        for key in self.free_parameter_keys:
            stride = self[key].free_values.size
            self[key].free_values = vector[i:(i + stride)]

            # Update repeated measurements
            self[key].update_repeated_values()

            i += stride

    def get_free_parameter_idx(self, parameter: str, pulse_id: int) -> int:
        """
        Allows you to get the index of a parameter pulse number combination in the free parameter vector.

        :param parameter: The name of free acquisition parameter
        :param pulse_id: The pulse for which you need the index
        :return: The index
        :raise KeyError: Parameter not found.
        """
        i = 0
        for key in self.free_parameter_keys:
            if key == parameter:
                return i + pulse_id
            stride = self[key].free_values.size
            i += stride
        raise KeyError(f"Parameter {parameter} not found")

    def set_free_parameter_bounds(self, bounds: List[Tuple[float, float]]) -> None:
        """
        Setter function for the parameter bounds, requires you to provide bounds for all parameters in sequence.

        :param bounds: The bounds
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
        return np.repeat([p.scale for p in self.values()], [p.free_values.size for p in self.values()])

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
        first_parameter = list(self.values())[0]
        return first_parameter.values.size

    def get_parameter_from_parameter_vector(self, parameter: str, x: np.ndarray) -> np.ndarray:
        """
        This function helps to get a free parameter from a numpy array. Useful for building constraints on specific
        parameters. Be warned the function does not check if the parameter you require is actually a free parameter!!

        :param parameter: The name of the free parameter you wish to extract from the array.
        :param x: A numpy array of free parameters, in scaled units.
        :return: The rescaled parameter values from the array (so now in physical units).
        :raise KeyError: Parameter not found.
        """
        i = 0
        for key in self.free_parameter_keys:
            stride = self[key].free_values.size
            if parameter == key:
                return x[i:(i + stride)] * self[key].scale
            i += stride
        raise KeyError(f"Parameter {parameter} not found")

    @property
    @abstractmethod
    def constraints(self) -> Dict[str, ConstraintTypes]:
        """
        Returns optimisation constraints on the scheme parameters. Implementation is child-class specific.

        :return: A scipy.optimize.LinearConstraint object. None is used to specify no constraints.
        The constraint is defined by lb <= A.x <= ub, x being the array of parameters optimized.
        A is the matrix defining the constraint relation between parameters.
        """
        raise NotImplementedError()

    @property
    def constraint_list(self) -> List[ConstraintTypes]:
        return list(self.constraints.values())

    @property
    def x0(self):
        scales = self.free_parameter_scales
        vector = self.free_parameter_vector
        return vector / scales

    def _copy_and_update_parameter(self, parameter: str, x: np.ndarray):
        """
        Makes a copy from Acquisition parameter values and inserts the values suggested by the optimizer in
        x. This allows us to extract the parameter values for all pulses in a single array also if the parameter was
        fixed for a particular pulse.

        :param parameter: The name of the parameter for which you want the update full value array
        :param x: The optimizer suggestion for ALL free parameters
        :return: The copy of all values and the updated free parameter values
        """
        if self[parameter].fixed:
            return copy(self[parameter].values)

        update_values = self.get_parameter_from_parameter_vector(parameter, x)
        value_array = copy(self[parameter].values)
        value_array[self[parameter].optimize_mask] = update_values
        return value_array

    def _are_fixed(self, parameter_list: List[str]) -> bool:
        return all(self[name].fixed for name in parameter_list)


# TODO: Revised until this line
# TODO: Use DiffusionAcquisitionScheme as a base-class for all other diffusion acquisition schemes.
class DiffusionAcquisitionScheme(AcquisitionScheme):
    """
    Defines a diffusion MR acquisition scheme.

    :param gradient_magnitudes: The gradient magnitudes in tesla per meter or equivalently mT/mm
    :param gradient_directions: A list or numpy array of direction cosines.
    :param pulse_widths: A list or numpy array of pulse widths δ in seconds.
    :param pulse_intervals: A list or numpy array of pulse intervals Δ in seconds.
    :param echo_times: A list or numpy array of the echo times in seconds.
    :param scan_parameters:  A ScannerParameters object that contains the quantities determined by scanner hardware
    :raise ValueError: b-vectors are not unit vectors or lists have unequal length.
    """

    def __init__(self,
                 gradient_magnitudes: Union[List[float], np.ndarray],
                 gradient_directions: Union[List[Tuple[float, float, float]], np.ndarray],
                 pulse_widths: Union[List[float], np.ndarray],
                 pulse_intervals: Union[List[float], np.ndarray],
                 echo_times: Optional[Union[List[float], np.ndarray]] = None,
                 scan_parameters: ScannerParameters = default_scanner):

        self.scan_parameters = scan_parameters

        b_values = get_b_value_complete(GAMMA, gradient_magnitudes, pulse_intervals, pulse_widths, scan_parameters)
        # set default echo times to minimal echo time based on scan parameters and b values
        if echo_times is None:
            # offset to prevent being to close to actual minimal echo times
            offset = 1e-6
            echo_times = minimal_echo_time(b_values, scan_parameters) + offset

        # converting to np array of correct type
        gradient_directions = np.asarray(gradient_directions, dtype=np.float64)

        # Calculate the spherical angles φ and θ.
        theta, phi = unitvector_to_angles(gradient_directions).T

        # checking for b0 values and setting vector to zero for these values
        self._check_b_vectors(b_values, gradient_directions)

        super().__init__({
            'DiffusionPulseMagnitude': AcquisitionParameters(
                values=gradient_magnitudes, unit=GRADIENT_UNIT, scale=1., symbol="|G|", lower_bound=0.0, upper_bound=5e3
            ),
            'DiffusionGradientAnglePhi': AcquisitionParameters(
                values=phi, unit='rad', scale=1., symbol=r"$\phi$", lower_bound=None, fixed=True
            ),
            'DiffusionGradientAngleTheta': AcquisitionParameters(
                values=theta, unit='rad', scale=1., symbol=r"$\theta$", lower_bound=None, fixed=True
            ),
            'DiffusionPulseWidth': AcquisitionParameters(
                values=pulse_widths, unit=PULSE_TIMING_UNIT, scale=PULSE_TIMING_SCALE, symbol=r"$\delta$", fixed=False,
                lower_bound=PULSE_TIMING_LB,
                upper_bound=PULSE_TIMING_UB
            ),
            'DiffusionPulseInterval': AcquisitionParameters(
                values=pulse_intervals, unit=PULSE_TIMING_UNIT, scale=PULSE_TIMING_SCALE, symbol=r"$\Delta$",
                fixed=False,
                lower_bound=PULSE_TIMING_LB,
                upper_bound=PULSE_TIMING_UB
            ),
            'EchoTime': AcquisitionParameters(
                values=echo_times, unit=PULSE_TIMING_UNIT, scale=PULSE_TIMING_SCALE, symbol=r"$T_E$", fixed=False,
                lower_bound=PULSE_TIMING_LB,
                upper_bound=PULSE_TIMING_UB
            )
        })

        self._check_constraints()

    @classmethod
    def from_bvals(cls, b_values: np.ndarray, b_vectors: np.ndarray, pulse_widths: np.ndarray,
                   pulse_intervals: np.ndarray,
                   echo_times: Optional[Union[List[float], np.ndarray]] = None,
                   scan_parameters: ScannerParameters = default_scanner):
        """
        Converts parameters and passes them to the main constructor __init__. See this constructor for more details.

        :param b_values: A list or numpy array of b-values in s/mm².
        :param b_vectors: A list or numpy array of direction cosines.
        :param pulse_widths: A list or numpy array of pulse widths δ in seconds.
        :param pulse_intervals: A list or numpy array of pulse intervals Δ in seconds.
        :param echo_times: A list or numpy array of the echo times in seconds.
        :param scan_parameters:  A ScannerParameters object that contains the quantities determined by scanner hardware
        :raise ValueError: b-vectors are not unit vectors or lists have unequal length.
        :return: DiffusionAcquisitionScheme
        """

        # convert bvals to pulse magnitudes
        gradient_magnitude = get_gradients(GAMMA, b_values, pulse_intervals, pulse_widths, scan_parameters)

        return cls(gradient_magnitude, b_vectors, pulse_widths, pulse_intervals, echo_times, scan_parameters)

    @staticmethod
    def _check_b_vectors(b_values: np.ndarray, b_vectors: np.ndarray) -> None:

        # Checking for b0 measurements
        b0 = b_values == 0

        # Checking for unit vectors
        if not np.allclose(np.linalg.norm(b_vectors[~b0], axis=1), 1):
            raise ValueError('b-vectors are not unit vectors.')

        # setting the vectors to (0,0,0) for the b0 measurements
        b_vectors[b0] = 0

    def _check_constraints(self):
        for desc, constraint in self.constraints.items():
            fun_val = constraint.fun(self.x0)

            lower_than_lb = is_smaller_than_with_tolerance(fun_val, constraint.lb)
            higher_than_ub = is_higher_than_with_tolerance(fun_val, constraint.ub)
            if lower_than_lb.any() or higher_than_ub.any():
                raise ValueError(f"DiffusionAcquisitionScheme: constraint violated, scheme does not satisfy {desc}")

    @property
    def b_values(self) -> np.ndarray:
        """
        An array of N b-values in s/mm².
        """
        return get_b_value_complete(GAMMA, self.pulse_magnitude, self.pulse_intervals, self.pulse_widths,
                                    self.scan_parameters)

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
        An array of N pulse widths in seconds.
        """
        return self['DiffusionPulseWidth'].values

    @property
    def pulse_intervals(self) -> np.ndarray:
        """
        An array of N pulse intervals in seconds.
        """
        return self['DiffusionPulseInterval'].values

    # TODO: verify results.
    @property
    def pulse_magnitude(self) -> np.ndarray:
        """
        Array of pulse magnitudes in mT/m
        """
        return self['DiffusionPulseMagnitude'].values

    @property
    def b_vectors(self) -> np.ndarray:
        """
        An N×3 array of direction cosines.
        """
        b_vectors = angles_to_unitvectors(np.array([self.theta, self.phi]).T)

        # Set b=0 'vectors' to (0, 0, 0) as per convention.
        b_vectors[self.b_values == 0] = 0

        return b_vectors

    @property
    def echo_times(self) -> np.ndarray:
        return self["EchoTime"].values

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

    @property
    def constraints(self) -> Optional[Dict[str, NonlinearConstraint]]:

        constraints = {}

        # Defining Δ > δ + epsilon + t180 or equivalently 0 < Δ - (δ + epsilon + t180) < \infty
        # We check in case both parameters are fixed we need not apply a constraint
        if not self._are_fixed(['DiffusionPulseWidth', 'DiffusionPulseInterval']):
            # get fixed parameter values and mask with the free parameter mask
            def delta_constraint_fun(x: np.ndarray):
                """ Should be larger than zero """
                pulse_width = self._copy_and_update_parameter('DiffusionPulseWidth', x)
                pulse_interval = self._copy_and_update_parameter('DiffusionPulseInterval', x)
                t_rise = self.scan_parameters.t_rise
                t_180 = self.scan_parameters.t_180
                return pulse_interval - (pulse_width + t_rise + t_180)

            delta_constraint = NonlinearConstraint(delta_constraint_fun, 0.0, np.inf, keep_feasible=True)
            constraints["PulseIntervalLargerThanPulseWidth"] = delta_constraint

        if not self._are_fixed(['DiffusionPulseMagnitude']):
            def gradient_constraint_fun(x: np.ndarray):
                g = self._copy_and_update_parameter('DiffusionPulseMagnitude', x)
                g_max = self.scan_parameters.G_max
                return g_max - g

            g_constraint = NonlinearConstraint(gradient_constraint_fun, 0.0, np.inf, keep_feasible=True)
            constraints["PulseMagnitudeSmallerThanMaxMagnitude"] = g_constraint

        if not self._are_fixed(['DiffusionPulseWidth', 'DiffusionPulseInterval', 'EchoTime']):
            def echo_constraint_fun(x: np.ndarray):
                self.set_free_parameter_vector(x * self.free_parameter_scales)
                echo_time = self._copy_and_update_parameter("EchoTime", x)
                t_min = minimal_echo_time(self.b_values, self.scan_parameters)

                return echo_time - t_min

            echo_time_constraint = NonlinearConstraint(echo_constraint_fun, 0.0, np.inf, keep_feasible=True)
            constraints["EchoTimeLargerThanMinEchoTime"] = echo_time_constraint

        return constraints

    def fix_b0_measurements(self) -> None:
        """
        Fixes the b0 values so they are not optimised. This is a utility method for using diffusion acquisition
        schemes with the dmipy package.

        :return:
        """
        b0_mask = self.b_values == 0

        for par in ["DiffusionPulseMagnitude", "DiffusionPulseWidth", "DiffusionPulseInterval", "EchoTime"]:
            # we should get the old fixed measurements and make sure that the new mask includes them
            old_mask = ~self[par].optimize_mask
            new_mask = np.logical_or(b0_mask, old_mask)
            self[par].set_fixed_mask(new_mask)


class DiffusionAcquisitionScheme_delta_dependency(AcquisitionScheme):
    """
    Defines a diffusion MR acquisition scheme.

    :param gradient_directions: A list or numpy array of direction cosines.
    :param echo_times: A list or numpy array of the echo times in seconds.
    :param scan_parameters:  A ScannerParameters object that contains the quantities determined by scanner hardware
    :raise ValueError: b-vectors are not unit vectors or lists have unequal length.
    """
            
    _required_parameters = ['gradient_magnitudes', 'gradient_directions', 'pulse_widths', 
                            'pulse_intervals', 'echo_times']

    def __init__(self,
                 gradient_magnitudes: Union[List[float], np.ndarray],
                 gradient_directions: Union[List[Tuple[float, float, float]], np.ndarray],
                 pulse_widths: Union[List[float], np.ndarray],
                 pulse_intervals: Union[List[float], np.ndarray],
                 echo_times: Optional[Union[List[float], np.ndarray]] = None,
                 scan_parameters: ScannerParameters = default_scanner):


        self.scan_parameters = scan_parameters   
        
        # converting to np array of correct type
        gradient_directions = np.asarray(gradient_directions, dtype=np.float64)
        # Calculate the spherical angles φ and θ.
        theta, phi = unitvector_to_angles(gradient_directions).T

        b_values = get_b_value_complete(GAMMA, gradient_magnitudes, pulse_intervals, pulse_widths, scan_parameters)
        
        if echo_times is None:
            # # offset to prevent being to close to actual minimal echo times
            offset = 1e-6
            min_echo_time = New_minimal_echo_time(scan_parameters) + offset
            echo_times = np.random.uniform(min_echo_time, MAX_TE, size=len(b_values))

        # checking for b0 values and setting vector to zero for these values
        self._check_b_vectors(b_values, gradient_directions)

        super().__init__({
            #Cristina 07-05
            'DiffusionPulseMagnitude': AcquisitionParameters(
                values=gradient_magnitudes, unit=GRADIENT_UNIT, scale=1., symbol="|G|", 
                lower_bound=0.0, upper_bound = scan_parameters.G_max
            ),
            'DiffusionGradientAnglePhi': AcquisitionParameters(
                values=phi, unit='rad', scale=1., symbol=r"$\phi$", lower_bound=None, fixed=True
            ),
            'DiffusionGradientAngleTheta': AcquisitionParameters(
                values=theta, unit='rad', scale=1., symbol=r"$\theta$", lower_bound=None, fixed=True
            ),
            'DiffusionPulseWidth': AcquisitionParameters(
                values=pulse_widths, unit=PULSE_TIMING_UNIT, scale=PULSE_TIMING_SCALE, symbol=r"$\delta$", fixed=False,
                lower_bound=PULSE_TIMING_LB,
                upper_bound=PULSE_TIMING_UB
            ),
            'DiffusionPulseInterval': AcquisitionParameters(
                values=pulse_intervals, unit=PULSE_TIMING_UNIT, scale=PULSE_TIMING_SCALE, symbol=r"$\Delta$",
                fixed=False,
                lower_bound=PULSE_TIMING_LB,
                upper_bound=PULSE_TIMING_UB
            ),
            'EchoTime': AcquisitionParameters(
                values=echo_times, unit=PULSE_TIMING_UNIT, scale=PULSE_TIMING_SCALE, symbol=r"$T_E$", fixed=False,
                lower_bound=New_minimal_echo_time(self.scan_parameters),
                upper_bound=MAX_TE
            )
        })
        
        #Cristina 04-07
        self._check_constraints()
        
    @classmethod
    def random_shell_initialization(cls, n_shells: int, n_directions: int, model_dependencies: list,
                                    scan_parameters: ScannerParameters = default_scanner):
        """
        Defines random initialization of DiffusionAcquisitionScheme based on a series of pulse relations and 
        boundaries per parameter. 

        """
        #Cristina 21-06
        #Directions are only sampled once, not iterative
        # TODO: Shouldn't we sample n_directions and duplicate those for each shell?
        # TODO: And wouldn't it be better if we could have an increasing number (squared) of directions per b-value?
        gradient_directions = sample_uniform_half_sphere(n_shells * n_directions)
        
        #Randomize the class based on its initialization parameters and constraints established by the model
        random_scheme = random_parameter_definition(cls._required_parameters, model_dependencies, 
                                                    n_shells, n_directions, scan_parameters)
        
        return cls(gradient_magnitudes=random_scheme['gradient_magnitudes'],
                   gradient_directions=gradient_directions,
                   pulse_widths=random_scheme['pulse_widths'],
                   pulse_intervals=random_scheme['pulse_intervals'],
                   echo_times=random_scheme['echo_times'],
                   scan_parameters=scan_parameters)
    
    @staticmethod
    def _check_b_vectors(b_values: np.ndarray, b_vectors: np.ndarray) -> None:
        
        # Checking for b0 measurements
        b0 = b_values == 0
        
        # Checking for unit vectors
        if not np.allclose(np.linalg.norm(b_vectors[~b0], axis=1), 1):
            raise ValueError('b-vectors are not unit vectors.')

        # setting the vectors to (0,0,0) for the b0 measurements
        b_vectors[b0] = 0

    def _check_constraints(self):
        for desc, constraint in self.constraints.items():
            fun_val = constraint.fun(self.x0)

            lower_than_lb = is_smaller_than_with_tolerance(fun_val, constraint.lb)
            higher_than_ub = is_higher_than_with_tolerance(fun_val, constraint.ub)
            if lower_than_lb.any() or higher_than_ub.any():
                raise ValueError(f"DiffusionAcquisitionScheme: constraint violated, scheme does not satisfy {desc}")                      

    #Cristina 12-07
    @property
    def b_values(self) -> np.ndarray:
        """
        An array of N b-values in s/mm².
        """        
        #In this case, b-values are not directly obtained but computed from pulse relations
        b_values = get_b_value_complete(GAMMA, self.pulse_magnitude, self.pulse_intervals, self.pulse_widths,
                                        self.scan_parameters)
            
        return b_values
        
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
        # Calculate the spherical angles φ and θ.
        return self['DiffusionGradientAngleTheta'].values
 
    @property
    def pulse_widths(self) -> np.ndarray:
        """
        An array of N pulse widths in seconds.
        """
        pulse_widths = self['DiffusionPulseWidth'].values
            
        return pulse_widths


    @property
    def pulse_intervals(self) -> np.ndarray:
        """
        An array of N pulse intervals in seconds.
        """
        pulse_intervals = self['DiffusionPulseInterval']
        
        return pulse_intervals

    @property
    def b_vectors(self) -> np.ndarray:
        """
        An N×3 array of direction cosines.
        """
        b_vectors = angles_to_unitvectors(np.array([self.theta, self.phi]).T)
    
        # Set b=0 'vectors' to (0, 0, 0) as per convention.
        b_vectors[self.b_values == 0] = 0

        return b_vectors

    @property
    def echo_times(self) -> np.ndarray:
        return self["EchoTime"].values

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
    
    @property
    def constraints(self) -> Optional[Dict[str, NonlinearConstraint]]:

        constraints = {}
            
        def delta_constraint_fun(x: np.ndarray):
            """ Should be larger than zero """
            
            #In this case deltas are optimize and used directly to compute b_val
            pulse_widths = self._copy_and_update_parameter('DiffusionPulseWidth', x)
            pulse_intervals = self._copy_and_update_parameter('DiffusionPulseInterval', x)
            
            t_rise = self.scan_parameters._t_rise*1e3
            t_180 = self.scan_parameters.t_180*1e3
            
            return pulse_intervals - (pulse_widths + t_rise + t_180)

        delta_constraint = NonlinearConstraint(delta_constraint_fun, 0.0, np.inf, keep_feasible=True)
        constraints["PulseIntervalLargerThanPulseWidth"] = delta_constraint

        def echo_constraint_fun(x: np.ndarray):

            echo_time = self._copy_and_update_parameter("EchoTime", x)
            
            t_min = New_minimal_echo_time(self.scan_parameters)
            
            return echo_time - t_min

        echo_time_constraint = NonlinearConstraint(echo_constraint_fun, 0.0, np.inf, keep_feasible=True)
        constraints["EchoTimeLargerThanMinEchoTime"] = echo_time_constraint
                               
        def gradient_b_val_constraint_fun(x: np.ndarray):
            
            g = self._copy_and_update_parameter('DiffusionPulseMagnitude', x)
            pulse_widths = self._copy_and_update_parameter('DiffusionPulseWidth', x)
            pulse_intervals = self._copy_and_update_parameter('DiffusionPulseInterval', x)               
            
            return B_MAX - b_val_from_delta_Delta(pulse_widths, pulse_intervals, g, self.scan_parameters)

        g_b_val_constraint = NonlinearConstraint(gradient_b_val_constraint_fun, 0.0, np.inf, keep_feasible=True)
        constraints["BValSmallerThanBMax"] = g_b_val_constraint
          
        return constraints
    
    def fix_b0_measurements(self) -> None:
        """
        Fixes the b0 values so they are not optimised. This is a utility method for using diffusion acquisition
        schemes with the dmipy package.

        :return:
        """
        #Cristina 12-07
        b0_mask = self.b_values == 0

        for par in ["DiffusionPulseMagnitude", "DiffusionPulseWidth", "DiffusionPulseInterval", "EchoTime"]:
            # we should get the old fixed measurements and make sure that the new mask includes them
            old_mask = ~self[par].optimize_mask
            new_mask = np.logical_or(b0_mask, old_mask)
            self[par].set_fixed_mask(new_mask)
            
class DiffusionAcquisitionScheme_bval_dependency(AcquisitionScheme):
    """
    Defines a diffusion MR acquisition scheme.

    :param gradient_directions: A list or numpy array of direction cosines.
    :param echo_times: A list or numpy array of the echo times in seconds.
    :param scan_parameters:  A ScannerParameters object that contains the quantities determined by scanner hardware
    :raise ValueError: b-vectors are not unit vectors or lists have unequal length.
    """

    _required_parameters = ['gradient_directions', 'echo_times', 'b_values']

    #Cristina 21-06: removed pulse width and interval from initialization: they will be computed from TE and b-val relation
    def __init__(self,
                 gradient_directions: Union[List[Tuple[float, float, float]], np.ndarray],
                 b_values: Union[List[float], np.ndarray],
                 echo_times: Optional[Union[List[float], np.ndarray]],
                 # model_dependencies: [List[str]],
                 scan_parameters: ScannerParameters = default_scanner):

        self.scan_parameters = scan_parameters
        # self.model_dependencies = model_dependencies

        # converting to np array of correct type
        gradient_directions = np.asarray(gradient_directions, dtype=np.float64)
        # Calculate the spherical angles φ and θ.
        theta, phi = unitvector_to_angles(gradient_directions).T

        # checking for b0 values and setting vector to zero for these values
        self._check_b_vectors(b_values, gradient_directions)

        super().__init__({
            'DiffusionGradientAnglePhi': AcquisitionParameters(
                values=phi, unit='rad', scale=1., symbol=r"$\phi$", lower_bound=None, fixed=True
            ),
            'DiffusionGradientAngleTheta': AcquisitionParameters(
                values=theta, unit='rad', scale=1., symbol=r"$\theta$", lower_bound=None, fixed=True
            ),
            # Cristina 09-05
            'EchoTime': AcquisitionParameters(
                values=echo_times, unit=PULSE_TIMING_UNIT, scale=PULSE_TIMING_SCALE, symbol=r"$T_E$", fixed=False,
                lower_bound=New_minimal_echo_time(self.scan_parameters),
                upper_bound=MAX_TE
            ),
            'B-Values': AcquisitionParameters(
                values=b_values, unit='s/mm²', scale=B_VAL_SCALE, symbol=r"$b$",
                lower_bound=B_VAL_LB, upper_bound=B_MAX
            ),
        })

        # Cristina 04-07
        self._check_constraints()

    @classmethod
    def random_shell_initialization(cls, n_shells: int, n_directions: int, model_dependencies: list,
                                    scan_parameters: ScannerParameters = default_scanner):
        """
        Defines random initialization of DiffusionAcquisitionScheme based on a series of pulse relations and
        boundaries per parameter.

        """
        # Cristina 21-06
        # Directions are only sampled once, not iterative
        # TODO: Shouldn't we sample n_directions and duplicate those for each shell?
        gradient_directions = sample_uniform_half_sphere(n_shells * n_directions)

        # Randomize the class based on its initialization parameters and constraints established by the model
        random_scheme = random_parameter_definition(cls._required_parameters, model_dependencies,
                                                    n_shells, n_directions, scan_parameters)

        return cls(gradient_directions=gradient_directions,
                   b_values=random_scheme['b_values'],
                   echo_times=random_scheme['echo_times'])

    @staticmethod
    def _check_b_vectors(b_values: np.ndarray, b_vectors: np.ndarray) -> None:

        # Checking for b0 measurements
        b0 = b_values == 0

        # Checking for unit vectors
        if not np.allclose(np.linalg.norm(b_vectors[~b0], axis=1), 1):
            raise ValueError('b-vectors are not unit vectors.')

        # setting the vectors to (0,0,0) for the b0 measurements
        b_vectors[b0] = 0

    def _check_constraints(self):
        for desc, constraint in self.constraints.items():
            fun_val = constraint.fun(self.x0)

            lower_than_lb = is_smaller_than_with_tolerance(fun_val, constraint.lb)
            higher_than_ub = is_higher_than_with_tolerance(fun_val, constraint.ub)
            if lower_than_lb.any() or higher_than_ub.any():
                raise ValueError(f"DiffusionAcquisitionScheme: constraint violated, scheme does not satisfy {desc}")

    #Cristina 12-07
    @property
    def b_values(self) -> np.ndarray:
        """
        An array of N b-values in s/mm².
        """
        return self['B-Values'].values

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
        An array of N pulse widths in seconds.
        """
        delta, _ = delta_Delta_from_TE(self.echo_times, self.scan_parameters)

        return delta

    @property
    def pulse_intervals(self) -> np.ndarray:
        """
        An array of N pulse intervals in seconds.
        """
        _, Delta = delta_Delta_from_TE(self.echo_times, self.scan_parameters)

        return Delta

    @property
    def b_vectors(self) -> np.ndarray:
        """
        An N×3 array of direction cosines.
        """
        b_vectors = angles_to_unitvectors(np.array([self.theta, self.phi]).T)

        # Set b=0 'vectors' to (0, 0, 0) as per convention.
        b_vectors[self.b_values == 0] = 0

        return b_vectors

    @property
    def echo_times(self) -> np.ndarray:
        return self["EchoTime"].values

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

    @property
    def constraints(self) -> Optional[Dict[str, NonlinearConstraint]]:

        constraints = {}

        def bval_TE_constraint_fun(x: np.ndarray):
            """ Should be larger than zero """

            b_values = self._copy_and_update_parameter('B-Values', x) #s/mm^2
            echo_times = self._copy_and_update_parameter("EchoTime", x) #[s]???

            #In this case compute deltas from optimized echo_time
            delta, Delta = delta_Delta_from_TE(echo_times, self.scan_parameters)

            #This computation is based on maximal G
            b_from_TE = b_val_from_delta_Delta(delta, Delta, self.scan_parameters.G_max, self.scan_parameters)

            return b_values - b_from_TE

        bval_TE_constraint = NonlinearConstraint(bval_TE_constraint_fun, - np.inf, 0.0, keep_feasible=True)
        constraints["BValueDependentOnTE"] = bval_TE_constraint

        return constraints

    def fix_b0_measurements(self) -> None:
        """
        Fixes the b0 values so they are not optimised. This is a utility method for using diffusion acquisition
        schemes with the dmipy package.

        :return:
        """
        #Cristina 12-07
        b0_mask = self.b_values == 0

        # for par in ["DiffusionPulseMagnitude", "DiffusionPulseWidth", "DiffusionPulseInterval", "EchoTime"]:
        for par in ["B-Values", "EchoTime"]:
            # we should get the old fixed measurements and make sure that the new mask includes them
            old_mask = ~self[par].optimize_mask
            new_mask = np.logical_or(b0_mask, old_mask)
            self[par].set_fixed_mask(new_mask)

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
            raise ValueError("Invalid inversion recovery scheme: at least one measurement breaks constrain TR>TE+TI")

        super().__init__(
            {
                'RepetitionTimeExcitation': AcquisitionParameters(
                    values=repetition_times, unit='ms', scale=100., symbol=r"$T_R$", lower_bound=10.0, upper_bound=1e4),
                'EchoTime': AcquisitionParameters(
                    values=echo_times, unit='ms', scale=10., symbol=r"$T_E$", fixed=True, lower_bound=.1,
                    upper_bound=1e3),
                'InversionTime': AcquisitionParameters(
                    values=inversion_times, unit='ms', scale=100., symbol=r"$T_I$", lower_bound=10.0, upper_bound=1e4)
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

    @property
    def constraints(self) -> Optional[Dict[str, ConstraintTypes]]:
        involved_parameters = ['InversionTime', 'EchoTime', 'RepetitionTimeExcitation']
        if self._are_fixed(involved_parameters):
            return None

        def time_constraint_fun(x: np.ndarray):
            # require return value larger than zero to enforce constraint TR > TE + TI
            ti = self._copy_and_update_parameter('InversionTime', x)
            te = self._copy_and_update_parameter('EchoTime', x)
            tr = self._copy_and_update_parameter('RepetitionTimeExcitation', x)
            return tr - te - ti

        return {"RepetitionTime_larger_than_Echotime_plus_InversionTime": NonlinearConstraint(time_constraint_fun, 0.0,
                                                                                              np.inf)}


class EchoScheme(AcquisitionScheme):
    def __init__(self, TE: np.ndarray):
        super().__init__({
            'EchoTime': AcquisitionParameters(values=TE, unit='ms', scale=1.0, symbol=r"$T_E$", lower_bound=1.0,
                                              upper_bound=200.0)
        })

    @property
    def echo_times(self):
        return self['EchoTime'].values

    @property
    def constraints(self) -> Optional[Union[dict, List[dict]]]:
        """ For now without constraints. """
        return None


class ReducedDiffusionScheme(AcquisitionScheme):
    def __init__(self, b_values: Union[List[float], np.ndarray], echo_times: Union[List[float], np.ndarray],
                 scanner_parameters: ScannerParameters = default_scanner
                 ):
        """
        :param b_values: the b values in s/mm²
        :param echo_times: The echo times in ms
        :param scanner_parameters: the parameters defined by the scanners settings and or hardware capabilities
        """
        self.scanner_parameters = scanner_parameters
        # Check for b0 values? make sure initial scheme satisfies constraints.

        super().__init__({
            'DiffusionBvalue': AcquisitionParameters(
                values=b_values, unit='s/mm²', scale=1000., symbol=r"$b$", lower_bound=0.0, upper_bound=3e4
            ),
            'EchoTime': AcquisitionParameters(
                values=echo_times, unit='ms', scale=10., symbol=r"$T_E$", lower_bound=0., upper_bound=1e3
            )
        })

    @property
    def echo_times(self):
        return self['EchoTime'].values

    @property
    def b_values(self):
        return self['DiffusionBvalue'].values

    @property
    def constraints(self) -> NonlinearConstraint:
        def fun(x: np.ndarray) -> np.ndarray:
            # get b-values from x
            b = self.get_parameter_from_parameter_vector('DiffusionBvalue', x)
            # note that b is in s/mm² but all other time dimensions are ms.
            # so we convert to ms/mm²
            b *= 1e3
            # get echotimes from x, (units are # ms)
            te = self.get_parameter_from_parameter_vector('EchoTime', x)
            # compute the minimal echotimes associated with b-values and other parameters
            te_min = minimal_echo_time(b, self.scanner_parameters)

            # The constraint is satisfied if actual TE is higher than minimal TE
            return te - te_min

        return NonlinearConstraint(fun, 0.0, np.inf)


def random_parameter_definition(required_params: List, randomization_constraints: List, n_shells: int,
                                n_directions: int, scan_parameters: default_scanner) -> dict:
    # to '1/mT . 1/ms'
    gamma = GAMMA * 1e-3
    
    # Get random values for n_shells and duplicate those for n_directions per shell.
    
    scheme_params = {}
    for param in required_params:
        # Gradient directions initialized once
        if param == 'gradient_directions':
            continue
        else:
            bounds = param_initialization_bounds(param)
            scheme_params[param] = np.random.uniform(bounds[0], bounds[1], size=n_shells)
  
    # Repeat randomization for values that do not follow the constraint
    while not np.all(constrained_dependencies(randomization_constraints, scheme_params, scan_parameters)):
        mask = ~constrained_dependencies(randomization_constraints, scheme_params, scan_parameters)        
        
        # Based on B-Value dependency uniquely
        if 'DiffusionPulseWidth' and 'DiffusionPulseInterval' not in randomization_constraints:
            scheme_params['b_values'][mask] = np.random.uniform(B_VAL_LB, B_VAL_UB, size=np.sum(mask))
            scheme_params['echo_times'][mask] = np.random.uniform(New_minimal_echo_time(default_scanner), MAX_TE, 
                                                                  size=np.sum(mask))
            
        elif 'DiffusionPulseWidth' and 'DiffusionPulseInterval' in randomization_constraints:
            
            # scheme_params['gradient_magnitudes'][mask] = np.random.uniform(0, default_scanner.G_max,
            #                                                     size=np.sum(mask))
            scheme_params['echo_times'][mask] = np.random.uniform(New_minimal_echo_time(default_scanner), MAX_TE,
                                                                  size=np.sum(mask))
            scheme_params['pulse_widths'][mask] = np.random.uniform(PULSE_TIMING_LB, PULSE_TIMING_UB, size=np.sum(mask))
            scheme_params['pulse_intervals'][mask] = np.random.uniform(PULSE_TIMING_LB, PULSE_TIMING_UB,
                                                                       size=np.sum(mask))
    
            scheme_params['gradient_magnitudes'][mask] = np.sqrt(
                B_VAL_UB / (
                    gamma**2 * (
                        scheme_params['pulse_widths'][mask]**2 *
                        (scheme_params['pulse_intervals'][mask] - scheme_params['pulse_widths'][mask] / 3) +
                        (scan_parameters.t_ramp**3) / 30 -
                        (scheme_params['pulse_widths'][mask] * scan_parameters.t_ramp**2) / 6)
                )
            )
        print('New randomization')
    
    # Duplicate the values for n_directions per shell.
    scheme_measurements = {key: np.repeat(value, n_directions) for key, value in scheme_params.items()}
            
    return scheme_measurements


def param_initialization_bounds(parameter: str):
        
    parameter_bound_dict = {
        'echo_times': [New_minimal_echo_time(default_scanner), MAX_TE],
        'b_values': [B_VAL_LB, B_VAL_UB],
        'pulse_intervals': [PULSE_TIMING_LB, PULSE_TIMING_UB],
        'pulse_widths': [PULSE_TIMING_LB, PULSE_TIMING_UB],
        'gradient_magnitudes': [0, 265e-3]
        }
    
    bounds = parameter_bound_dict[parameter]
    
    return bounds

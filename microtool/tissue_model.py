"""
The TissueModel is a base class for different kinds of tissue models (different types of models and/or different types
 of modelling-software).

In order to simulate the MR signal in response to a MICROtool acquisition scheme, the TissueModel first translates the
 acquisition scheme to a modelling-software specific one.
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Union, List, Optional, Sequence

import numpy as np
from numpy.random import default_rng
from scipy.optimize import minimize, Bounds, OptimizeResult, curve_fit, differential_evolution, LinearConstraint, \
    basinhopping
from tabulate import tabulate

from .acquisition_scheme import AcquisitionScheme, InversionRecoveryAcquisitionScheme, EchoScheme, \
    ReducedDiffusionScheme, DiffusionAcquisitionScheme, DiffusionAcquisitionScheme_bval_dependency, \
    DiffusionAcquisitionScheme_delta_dependency
from .constants import VOLUME_FRACTION_PREFIX, MODEL_PREFIX, BASE_SIGNAL_KEY, T2_KEY, T1_KEY, \
    DIFFUSIVITY_KEY, RELAXATION_BOUNDS, ConstraintTypes
from .utils.unit_registry import unit, cast_to_ndarray


@dataclass
class TissueParameter:
    # noinspection PyUnresolvedReferences
    """
    Defines a scalar tissue parameter and its properties.

    :param value: Parameter value.
    :param scale: The typical parameter value scale (order of magnitude).
    :param optimize: Specifies if the parameter should be included in the optimization of an AcquisitionScheme.
                    If we don't optimize the scheme we assume the parameter is known and exclude it from fitting too.
    :param fit_flag: Specifies if the parameter should be included in the fitting process.
    :param fit_bounds: Specifies the domain in which to attempt the fitting of this parameter.
    """
    value: float
    scale: float
    optimize: bool = True
    fit_flag: bool = True
    fit_bounds: tuple = (0.0, np.inf)

    def __post_init__(self):
        self.fit_guess = self.scale

    def __str__(self):
        return (f'{self.value} (scale: {self.scale}, optimize: {self.optimize}, fit:{self.fit_flag}, '
                f'bounds:{self.fit_bounds})')


class TissueModel(Dict[str, TissueParameter], ABC):
    """
    Base-class for Tissue Models.

    :param parameters: A dictionary with TissueParameter definitions. 
    """

    @abstractmethod
    def __call__(self, scheme: AcquisitionScheme) -> np.ndarray:
        """
        Calculates the MR signal attenuation.

        :param scheme: An AcquisitionScheme.
        :return: A array with N signal attenuation values, where N is the number of samples.
        """
        raise NotImplementedError()

    def __init__(self, parameters: Dict[str, TissueParameter]):
        super().__init__(parameters)

    def __str__(self) -> str:

        table = []
        for key, value in self.items():
            table.append([key, value.value, value.scale, value.optimize, value.fit_flag, value.fit_bounds])

        table_str = tabulate(table, headers=["Tissue-parameter", "Value", "Scale", "Optimize", "Fit", "Fit Bounds"])

        return f'Tissue model with {len(self)} scalar parameters:\n{table_str}'

    def _set_finite_difference_vars(self):
        """
        Defines parameter values in diagonal matrix to be utilized in jacobian computation
        """
        # Get the baseline parameter vector, but don't include S0.
        # Downcast pint-wrapped array (testing only) to a plain numpy array.
        self._parameter_baseline = self.scaled_parameter_vector

        # Calculate finite differences and corresponding parameter vectors for calculating derivatives.
        self._step_size = 1e-3
        h = self._step_size * np.identity(len(self))
        self._parameter_vectors_forward = self._parameter_baseline + 0.5 * h
        self._parameter_vectors_backward = self._parameter_baseline - 0.5 * h

    def scaled_jacobian(self, scheme: AcquisitionScheme) -> np.ndarray:
        """
        Calculates the change in MR signal attenuation due to a change in the (scaled) tissue model parameters.

        The order of the tissue parameters is the same as returned by get_parameters().

        :param scheme: An AcquisitionScheme.
        :return: An N×M scaled Jacobian matrix, where N is the number of samples and M is the number of tissue
         parameters.
        """
        self._set_finite_difference_vars()

        forward_diff = self._simulate_signals_scaled(self._parameter_vectors_forward, scheme)
        backward_diff = self._simulate_signals_scaled(self._parameter_vectors_backward, scheme)
        jac = (forward_diff - backward_diff) / self._step_size

        if np.all(jac == 0):
            raise RuntimeError("Jacobian is zero")

        # reset parameters to original
        self.set_scaled_parameters(self._parameter_baseline)

        # Return the relevant part of the Jacobian.
        return jac.T[:, self.include_optimize]

    def _simulate_signals_scaled(self, parameter_vectors: np.ndarray, scheme: AcquisitionScheme) -> np.ndarray:
        """
        Simulate a series of signals.
        
        :param parameter_vectors: numpy array with new set of N (scaled) parameter values to define the TissueModel with
        :param scheme: an AcquisitionScheme instance
        :return: Nx1 simulated signals from modified model with parameter(i) at each iteration i = 1:N 
        """
        original = self.scaled_parameter_vector

        npv = parameter_vectors.shape[0]  # number of parameter vectors
        signals = np.zeros((npv, scheme.pulse_count))
        for i in range(npv):
            self.set_scaled_parameters(parameter_vectors[i, :])
            signals[i, :] = self.__call__(scheme)

        self.set_scaled_parameters(original)

        return signals

    @abstractmethod
    def fit(self, scheme: AcquisitionScheme, signal: np.ndarray, **fit_options) -> FittedModel:
        """
        Fits the tissue model parameters to noisy_signal data given an acquisition scheme.
        :param signal: The noisy signal
        :param scheme: The scheme under investigation
        :return: A FittedTissueModel
        """
        raise NotImplementedError()

    @property
    def parameters(self) -> Dict[str, Union[float, np.ndarray]]:
        parameters = {}
        for name, parameter in self.items():
            parameters[name] = parameter.value
        return parameters

    @property
    def parameter_names(self) -> List[str]:
        return [key for key in self.keys()]

    def set_scaled_parameters(self, new_parameter_values: Sequence[float]) -> None:
        """
        Parameter values of TissueModel are redefined based on new array
        
        Note: You should probably overwrite this method if you are using a wrapped model.

        :param new_parameter_values: new array of values to define parameters of a TissueModel
        """
        for parameter, new_value in zip(self.values(), new_parameter_values):
            parameter.value = new_value * parameter.scale

    def set_scaled_fit_parameters(self, new_values: np.ndarray) -> None:
        """
        Sets the TissueParameters that are flagged for fitting to the provided values.

        :param new_values: The parameter values to be set as a numpy array.
        :return: None
        """
        if np.sum(self.include_fit) != new_values.shape[0]:
            raise ValueError("Shape of new values does not match number of parameters marked for fitting.")

        i = 0
        for parameter in self.values():
            if parameter.fit_flag:
                parameter.value = new_values[i] * parameter.scale
                i += 1

    @property
    def scaled_parameter_vector(self) -> np.ndarray:
        # put all parameter values in a single array
        vector = np.array([parameter.value / parameter.scale for parameter in self.values()])
        return vector

    @property
    def include_optimize(self):
        return np.array([value.optimize for value in self.values()])

    @property
    def include_fit(self):
        return np.array([value.fit_flag for value in self.values()])

    @property
    def scaled_fit_bounds_all(self) -> Bounds:
        lbs = []
        ubs = []
        for parameter in self.values():
            if parameter.fit_flag:
                ub = parameter.fit_bounds[1] / parameter.scale
                lb = parameter.fit_bounds[0] / parameter.scale
                if lb is None:
                    lbs.append(-np.inf)
                else:
                    lbs.append(lb)
                if ub is None:
                    ubs.append(np.inf)
                else:
                    if ub < lb:
                        raise ValueError("Upper bound is larger than lower bound.")
                    ubs.append(ub)

        return Bounds(np.array(lbs), np.array(ubs), keep_feasible=False)

    @property
    def scaled_fit_initial_guess(self) -> np.ndarray:
        return np.array([parameter.fit_guess / parameter.scale for parameter in self.values() if parameter.fit_flag])

    def check_dependencies(self, scheme: AcquisitionScheme):
        """
        Method for consistency check-up between model requirements and defined scheme parameters

        """
        return NotImplementedError()

    def get_dependencies(self):
        """
        Retrieve scheme parameter dependencies based on defined model.
        """
        return NotImplementedError()


class MultiTissueModel(TissueModel):
    """
    Class for multi-comparment models.

    """
    def __init__(self, models: List[TissueModel], volume_fractions: Optional[List[float]] = None):

        self._models = models
        # making a parameter dictionary using parameters in the individual compartments
        parameters = {}
        param_location = {}
        param_index = 0
        for i, model in enumerate(models):
            for key, value in model.items():
                if key != BASE_SIGNAL_KEY:
                    parameters.update({f"{MODEL_PREFIX}{i}_{key}": value})
                    param_location.update({f"{MODEL_PREFIX}{i}_{key}": param_index})
                    param_index += 1
                else:
                    param_location.update({f"{MODEL_PREFIX}{i}_{key}": None})

        if len(self._models) > 1:
            if volume_fractions is None:
                raise ValueError("Please provide volume fractions if you include multiple models")
            # Inserting the partial volumes as model parameters
            if len(volume_fractions) != len(self._models):
                raise ValueError("Not enough volume fractions provided for number of models")
            if sum(volume_fractions) != 1.:
                raise ValueError("Volume fractions dont sum to 1")

            # The 0th volume fractions is defined as 1 - the others so we mark it to be excluded from fitting
            for i, vf in enumerate(volume_fractions):
                parameters.update({
                    f"{VOLUME_FRACTION_PREFIX}{i}": TissueParameter(value=vf, scale=1., fit_bounds=(0.0, 1.0),
                                                                    fit_flag=False if i == 0 else True)
                })
                param_location.update({f"{VOLUME_FRACTION_PREFIX}{i}": param_index})
                param_index += 1

        # Add S0 as a tissue parameter (to be excluded in parameters extraction etc.)
        parameters.update({BASE_SIGNAL_KEY: TissueParameter(value=1.0, scale=1.0, optimize=False, fit_flag=False,
                                                            fit_bounds=(0.0, 2.0))})
        param_location.update({f"{MODEL_PREFIX}{BASE_SIGNAL_KEY}": param_index})
        self._param_location = param_location

        super().__init__(parameters)

    def set_scaled_parameters(self, new_parameter_values: Sequence[float]) -> None:
        """
        New definition of parameter values performed initially on each individual compartment and later
        on MultiTissue instance to ensure consistency
        """ 
        # Make sure that the parameters are updated on the individual models first
        for j, model in enumerate(self._models):
            parameter_update = []
            n_p = len(model)  # Model length for check-up
            
            # Obtain original parameters from new MultiTissue instance for each model
            for key in model: 
                index_param = self._param_location.get(f"{MODEL_PREFIX}{j}_{key}")
                if key == 'S0' and index_param is None:
                    # param_value = 1
                    parameter_update.append(1)
                elif key != 'S0' and index_param is None:
                    raise ValueError(f"Parameter {key} for {MODEL_PREFIX}{j} update is missing")
                else:
                    parameter_update.append(new_parameter_values[index_param])                 

            # Check length of parameter update matches length of model
            # Note: Error should never be reached as parameter check-up is performed in iterative process
            if len(parameter_update) != n_p:
                raise ValueError("Missing parameters for model update")
            else:
                model.set_scaled_parameters(np.array(parameter_update))
                 
        # Update MultiTissue instance lastly
        super().set_scaled_parameters(new_parameter_values)

    def set_scaled_fit_parameters(self, new_values: np.ndarray) -> None:
        """
        New definition of parameter values for fitting both in compartment models and MultiTissueModel
        """ 
        super().set_scaled_fit_parameters(new_values)

        if isinstance(new_values, dict):
            new_values = np.array(list(new_values.values()))

        # We also update the parameters on the models in this object
        for j, model in enumerate(self._models):
            parameter_update = []            
            
            for key in model: 
                index_param = self._param_location.get(f"{MODEL_PREFIX}{j}_{key}")
                
                if key == 'S0' and index_param is None:
                    continue
                elif key != 'S0' and index_param is None:
                    raise ValueError(f"Parameter {key} for {MODEL_PREFIX}{j} update is missing")
                else: 
                    if self.include_fit[index_param]:
                        parameter_update.append(new_values[index_param])
             
            model.set_scaled_fit_parameters(np.array(parameter_update))
        
        # update volume fraction 0 if necessary
        if len(self._models) > 1:
            self[VOLUME_FRACTION_PREFIX + "0"].value = 1 - np.sum(self.volume_fractions[1:])

    def __call__(self, scheme: AcquisitionScheme) -> np.ndarray:
        # use the call functions of the models
        compartment_signals = np.stack([model(scheme) for model in self._models], axis=-1)
        return np.sum(compartment_signals * self.volume_fractions, axis=-1)

    def check_dependencies(self, scheme: AcquisitionScheme):        
        """
        Method for consistency check-up between model requirements and defined scheme parameters
        """        
        # Check model-specific requirements
        for i in range(len(self._models)):
            model = self._models[i]          
            model.check_dependencies(scheme)            
 
    def fit(self, scheme: AcquisitionScheme, signal: np.ndarray, method: Union[str, callable] = 'trust-constr',
            **fit_options) -> FittedModelMinimize:
        """
        Fits the tissue model parameters to noisy_signal data given an acquisition scheme.

        :param scheme: The scheme under investigation
        :param signal: The noisy signal
        :param method: Type of solver. Either 'differential_evolution', 'basinhopping' or a solver available in
         scipy.optimize.minimize.
        :return: A FittedModelMinimize
        """

        cost_fun_args = (signal, scheme, deepcopy(self))

        # scaling the bounds
        bounds = self.scaled_fit_bounds_all

        x0 = self.scaled_fit_initial_guess
        minimize_kwargs = {"args": cost_fun_args, "bounds": bounds, "constraints": self.fit_constraints}
        if method == 'differential_evolution':
            result = differential_evolution(fit_cost,
                                            **minimize_kwargs,
                                            workers=-1,
                                            updating='deferred',
                                            disp=True)
        elif method == "basinhopping":
            minimize_kwargs.update({"method": "trust-constr"})
            result = basinhopping(fit_cost, x0, minimizer_kwargs=minimize_kwargs, disp=True)
        else:
            rng = default_rng()
            machine_epsilon = np.finfo(float).eps
            n_init = 10
            best_cost = np.inf
            best_result = None
            for _ in range(n_init):
                # Generate initial guess in bounds where we account for machine precision to prevent stepping out

                x0 = rng.uniform(low=bounds.lb + machine_epsilon, high=bounds.ub - machine_epsilon, size=bounds.lb.size)

                result = minimize(fit_cost, x0=x0, **minimize_kwargs,
                                  method=method, options={})

                if result.fun < best_cost:
                    best_cost = result.fun
                    best_result = result

            result = best_result

        result.x = result.x
        return FittedModelMinimize(self, result)

    @property
    def fit_constraints(self) -> ConstraintTypes:
        if len(self._models) == 1:
            return []

        # for now only volume fractions.
        mat = []
        for name in np.array(self.parameter_names)[self.include_fit]:
            if name.startswith(VOLUME_FRACTION_PREFIX):
                mat.append(1)
            else:
                mat.append(0)
        return LinearConstraint(mat, 0, 1, keep_feasible=False)

    @property
    def volume_fractions(self) -> np.ndarray:
        """
        An array with volume fractions definining the multi-compartment model and unit fraction for 
        each compartment
        """
        vfs = []
        for key, parameter in self.items():
            if key.startswith(VOLUME_FRACTION_PREFIX):
                vfs.append(parameter.value)

        # if no volume fractions are present we are dealing with single model so return 1.0
        if len(vfs) == 0:
            vfs.append(1.0)

        return np.array(vfs)


class RelaxationTissueModel(TissueModel):
    """
    Defines tissue by its relaxation parameters.

    :param t1: Longitudinal relaxation time constant T1 in seconds.
    :param t2: Transverse relaxation time constant T2 in seconds.
    """

    def __init__(self, model: TissueModel, t2: float, t1: Optional[float] = None):
        
        self._model = model
        base_signal = self._model[BASE_SIGNAL_KEY].value
           
        parameters = {MODEL_PREFIX + key: value for key, value in model.items() if key != BASE_SIGNAL_KEY}

        if t2 is None:
            raise ValueError("Expected T2 relaxation values.")
        elif t1 is None:
            parameters.update({T2_KEY: TissueParameter(value=t2, scale=1e-3 * unit('s'), optimize=True,
                                                       fit_flag=True, fit_bounds=RELAXATION_BOUNDS)})
        else:
            parameters.update({T2_KEY: TissueParameter(value=t2, scale=1e-3 * unit('s'), optimize=True),
                              T1_KEY: TissueParameter(value=t1, scale=1e-3 * unit('s'), optimize=False)})
               
        parameters.update({BASE_SIGNAL_KEY: TissueParameter(value=base_signal, scale=1.0, optimize=False, 
                                                            fit_flag=False, fit_bounds=(0.0, 2.0))})
        
        super().__init__(parameters)
                
    def __call__(self, scheme: AcquisitionScheme) -> np.ndarray:
        # Signal from tissue model
        if type(self._model) is TissueModel:
            model_signal = self[BASE_SIGNAL_KEY].value
        else:
            model_signal = self._model(scheme)
     
        if isinstance(scheme, (DiffusionAcquisitionScheme, 
                               DiffusionAcquisitionScheme_bval_dependency, 
                               DiffusionAcquisitionScheme_delta_dependency)):
            te = scheme.echo_times  # [s]
            t2 = self[T2_KEY].value
            te_t2 = np.exp(- te / t2)
            
            signal = model_signal * te_t2  # S0 * exp(-TE/T2)
        
        elif isinstance(scheme, InversionRecoveryAcquisitionScheme):
            if self['T1'] is None:
                raise ValueError("Expected T1 values for Inversion Recovery scheme")
            ti = scheme.inversion_times  # [s]
            tr = scheme.repetition_times  # [s]
            te = scheme.echo_times  # [s]

            ti_t1 = np.exp(-ti / self[T1_KEY].value)
            tr_t1 = np.exp(-tr / self[T1_KEY].value)
            te_t2 = np.exp(-te / self[T2_KEY].value)

            signal = model_signal * (1 - 2 * ti_t1 + tr_t1) * te_t2

        else:
            raise ValueError("Unsupported scheme type")
                         
        return signal
   
    def get_dependencies(self):       
        """
        Method to retrieve scheme parameter dependencies based on defined model. 
        Currently only implemented for DmipyTissueModels
        """
        
        dependencies = self._model.get_dependencies()
            
        return dependencies

    def check_dependencies(self, scheme: AcquisitionScheme):        
        """
        Method for consistency check-up between model requirements and defined scheme parameters

        """  
        # If T2 is not utilized for fitting
        if not self['T2'].fit_flag:    
            if not scheme['EchoTime'].fixed:
                warnings.warn("If T2 relaxation is not used for fitting, echo time should be fixed.")
        else:
            if scheme['EchoTime'].fixed:
                warnings.warn("If T2 relaxation used for fitting, echo time should not be fixed.")

    def fit(self, scheme: AcquisitionScheme, signal: np.ndarray, **fit_options) -> FittedModel:
        # TODO: Implement this?
        raise NotImplementedError


class ExponentialTissueModel(TissueModel):
    def __init__(self, t2: float, s0: float = 1.0):
        """

        :param t2: The T2 relaxation in [s]
        :param s0: The initial signal
        """
        super().__init__({
            T2_KEY: TissueParameter(value=t2, scale=1e-3 * unit('s'), optimize=True,
                                    fit_bounds=(.1e-3 * unit('s'), 10 * unit('s'))),
            BASE_SIGNAL_KEY: TissueParameter(value=s0, scale=1.0, optimize=True, fit_bounds=(.1, 2))
        })

    def __call__(self, scheme: EchoScheme) -> np.ndarray:
        """
        Implement the signal equation S = S0 * exp(-TE/T2) here
        :return:
        """
        te = scheme.echo_times
        t2 = self[T2_KEY].value
        s0 = self[BASE_SIGNAL_KEY].value
        return s0 * np.exp(- te / t2)

    def scaled_jacobian(self, scheme: EchoScheme) -> np.ndarray:
        """
        This is the analytical way of computing the jacobian.
        :param scheme:
        :return:
        """
        te = scheme.echo_times
        t2 = self[T2_KEY].value
        s0 = self[BASE_SIGNAL_KEY].value

        # the base signal
        s = s0 * np.exp(-te / t2)
        # return np.array([-TE * S, 1]).T.
        jac = cast_to_ndarray([(te / t2 ** 2) * self[T2_KEY].scale, s / s0 * self[BASE_SIGNAL_KEY].scale]).T
        return jac[:, self.include_optimize]

    def fit(self, scheme: EchoScheme, signal: np.ndarray, **fit_options) -> FittedModelCurveFit:
        """

        :param scheme:
        :param signal:
        :param fit_options:
        :return:
        """
        # extracting the echo times from the scheme
        te = scheme.echo_times

        # Whether or not to include a parameter in the fitting process?
        include = self.include_optimize

        # initial parameters in case we want to exclude some parameter from the fitting process
        parameter_value = list(self.parameters.values())

        # the function defining the signal in form compatible with scipy curve fitting
        # TODO: Why don't we set the values and use __call__ here?
        def signal_fun(_measurement, t2, s0):
            # Re-scale the parameters to their original scale.
            t2 *= self[T2_KEY].scale
            s0 *= self[BASE_SIGNAL_KEY].scale

            if not include[0]:
                t2 = parameter_value[0]
            if not include[1]:
                s0 = parameter_value[1]

            return s0 * np.exp(-te / t2)

        bounds = self.scaled_fit_bounds_all
        result = curve_fit(signal_fun, np.arange(len(te)), signal, self.scaled_fit_initial_guess,
                           bounds=(bounds.lb, bounds.ub),
                           maxfev=4 ** 2 * 100, full_output=True, **fit_options)

        return FittedModelCurveFit(self, result)


class RelaxedIsotropicModel(TissueModel):
    def fit(self, scheme: AcquisitionScheme, signal: np.ndarray, **fit_options) -> FittedModelCurveFit:
        raise NotImplementedError()

    def __init__(self, t2: float, diffusivity: float, s0: float = 1.0):
        """
        :param t2: The tissues T2 in [s]
        :param diffusivity: The tissue diffusivity in [mm² / s]
        :param s0: signal at the zeroth measurement [dimensionless]
        """
        super().__init__({
            T2_KEY: TissueParameter(value=t2, scale=1e-3 * unit('s')),
            DIFFUSIVITY_KEY: TissueParameter(value=diffusivity, scale=diffusivity),
            BASE_SIGNAL_KEY: TissueParameter(value=s0, scale=s0, optimize=False),
        })

    def __call__(self, scheme: ReducedDiffusionScheme) -> np.ndarray:
        # The signal equation for this model S = S0*exp(-b.*D).*exp(-TE/T2)

        bvalues = scheme.b_values
        te = scheme.echo_times

        b_d = np.exp(-bvalues * self[DIFFUSIVITY_KEY].value)
        te_t2 = np.exp(- te / self[T2_KEY].value)
        return self[BASE_SIGNAL_KEY].value * b_d * te_t2

    def scaled_jacobian(self, scheme: ReducedDiffusionScheme) -> np.ndarray:
        # Acquisition parameters
        bvalues = scheme.b_values
        te = scheme.echo_times

        # tissuemodel parameters
        t2 = self[T2_KEY].value
        d = self[DIFFUSIVITY_KEY].value
        s0 = self[BASE_SIGNAL_KEY].value

        # Exponents
        b_d = np.exp(-bvalues * d)
        te_t2 = np.exp(- te / t2)

        jac = np.array([
            (te * s0 * b_d * te_t2 / t2 ** 2) * self[T2_KEY].scale,
            (-bvalues * s0 * b_d * te_t2) * self[DIFFUSIVITY_KEY].scale,
            (b_d * te_t2) * self[BASE_SIGNAL_KEY].scale]).T
        return jac[:, self.include_optimize]


class FittedModel(ABC):
    @abstractmethod
    def print_fit_information(self) -> None:
        raise NotImplementedError()

    @property
    @abstractmethod
    def fitted_parameters(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError()


class FittedModelCurveFit(FittedModel):
    def __init__(self, model: TissueModel, curve_fit_result: tuple):
        if len(curve_fit_result) != 5:
            raise ValueError("Expected a full output curve fit result.")

        # last parameter of curve fit result is a useless int flag
        optimal_pars, covariance_matrix, fit_information, message, _ = curve_fit_result
        fit_information.update({"covariance_matrix": covariance_matrix, "message": message})

        self._model = model
        self.fitted_parameters_vector = optimal_pars
        self._fit_information = fit_information

    @property
    def fitted_parameters(self) -> Dict[str, np.ndarray]:

        vector = self.fitted_parameters_vector
        parameter_names = self._model.parameter_names
        include = self._model.include_optimize

        out = {}
        for i in range(len(parameter_names)):
            if include[i]:
                out.update({parameter_names[i]: vector[i]})

        return out

    @property
    def print_fit_information(self) -> Optional[dict]:
        return self._fit_information


class FittedModelMinimize(FittedModel):
    def __init__(self, model: TissueModel, result: OptimizeResult):
        self.model = model
        self.result = result

        if not self.result.success:
            warnings.warn(
                "Minimize says optimization was unsuccessful inspect fit information to decide on further actions.",
                category=RuntimeWarning)

    def print_fit_information(self) -> None:
        for key in self.result.keys():
            print(key, self.result[key])

    @property
    def fitted_parameters(self) -> Dict[str, np.ndarray]:
        fit_parameters = {key: parameter for key, parameter in self.model.items() if parameter.fit_flag}

        return {key: value * parameter.scale for (key, parameter), value in zip(fit_parameters.items(), self.result.x)}

    @property
    def scaled_fitted_parameters(self) -> np.ndarray:
        return self.result.x


# TODO: Do we need this?
class TissueModelDecorator(TissueModel, ABC):
    """
    Abstract class for initialization of TissueModel decorators. this just passes all public methods to the original
    object override these methods to extend or alter functionality while retaining the same interface.

    This concept is based on the decorator design pattern and more information can be found at
    https://refactoring.guru/design-patterns/decorator
    """

    def __init__(self, original: TissueModel):
        self._original = deepcopy(original)

    def __call__(self, scheme: AcquisitionScheme) -> np.ndarray:
        return self._original.__call__(scheme)

    def __str__(self):
        return self._original.__str__()

    def scaled_jacobian(self, scheme: AcquisitionScheme) -> np.ndarray:
        return self._original.scaled_jacobian(scheme)

    def fit(self, scheme: AcquisitionScheme, signal: np.ndarray, **fit_options) -> FittedModel:
        return self._original.fit(scheme, signal, **fit_options)

    @property
    def parameters(self):
        return self._original.parameters

    @property
    def parameter_names(self):
        return self._original.parameter_names

    @property
    def include_optimize(self):
        return self._original.include_optimize

    @property
    def scaled_parameter_vector(self) -> np.ndarray:
        return self._original.scaled_parameter_vector


# TODO add docstrings
def fit_cost(fit_parameter_vector, signal, scheme, model: TissueModel):
    model.set_scaled_fit_parameters(fit_parameter_vector)
    predicted_signal = model(scheme)
    square_diff = (signal - predicted_signal) ** 2
    return np.sum(square_diff)

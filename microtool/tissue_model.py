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
from typing import Dict, Union, List, Optional

import numpy as np
from numpy.random import default_rng
from scipy.optimize import minimize, Bounds, OptimizeResult, curve_fit, differential_evolution, NonlinearConstraint, \
    LinearConstraint
from tabulate import tabulate

from .acquisition_scheme import AcquisitionScheme, InversionRecoveryAcquisitionScheme, EchoScheme, \
    ReducedDiffusionScheme
from .constants import VOLUME_FRACTION_PREFIX, MODEL_PREFIX, BASE_SIGNAL_KEY, RELAXATION_PREFIX, T2_KEY, T1_KEY, \
    DIFFUSIVITY_KEY, RELAXATION_BOUNDS

Constraints = Union[List[Union[LinearConstraint, NonlinearConstraint]], Union[LinearConstraint, NonlinearConstraint]]


@dataclass
class TissueParameter:
    # noinspection PyUnresolvedReferences
    """
    Defines a scalar tissue parameter and its properties.

    :param value: Parameter value.
    :param scale: The typical parameter value scale (order of magnitude).
    :param optimize: Specifies if the parameter should be included in the optimization of an AcquisitionScheme.
                    If we dont optimize the scheme we assume the parameter is known and exclude it from fitting as well.
    :param fit_flag: Specifies if the parameter should be included in the fitting process.
    :param fit_bounds: Specificies the domain in which to attempt the fitting of this parameter.
    """
    value: float
    scale: float
    optimize: bool = True
    fit_flag: bool = True
    fit_bounds: tuple = (0.0, np.inf)

    def __post_init__(self):
        self.fit_guess = self.scale * .5

    def __str__(self):
        return f'{self.value} (scale: {self.scale}, optimize: {self.optimize}, fit:{self.fit_flag},bounds:{self.fit_bounds})'


class TissueModel(Dict[str, TissueParameter], ABC):
    # step size for the finite difference method.

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
        self._set_finite_difference_vars()

    def __str__(self) -> str:

        table = []
        for key, value in self.items():
            table.append([key, value.value, value.scale, value.optimize, value.fit_flag, value.fit_bounds])

        table_str = tabulate(table, headers=["Tissue-parameter", "Value", "Scale", "Optimize", "Fit", "Fit Bounds"])

        return f'Tissue model with {len(self)} scalar parameters:\n{table_str}'

    def _set_finite_difference_vars(self):
        # Get the baseline parameter vector, but don't include S0.
        self._parameter_baseline = np.array([parameter.value for parameter in self.values()])[:-1]

        step_size = 1e-3
        # Calculate finite differences and corresponding parameter vectors for calculating derivatives.
        h = np.array([parameter.scale * step_size for parameter in self.values()])[:-1]
        self._parameter_vectors_forward = self._parameter_baseline + 0.5 * np.diag(h)
        self._parameter_vectors_backward = self._parameter_baseline - 0.5 * np.diag(h)
        self._reciprocal_h = (1 / h).reshape(-1, 1)

    def jacobian(self, scheme: AcquisitionScheme) -> np.ndarray:
        """
        Calculates the change in MR signal attenuation due to a change in the tissue model parameters.

        The order of the tissue parameters is the same as returned by get_parameters().

        :param scheme: An AcquisitionScheme.
        :return: An N×M Jacobian matrix, where N is the number of samples and M is the number of tissue parameters.
        """
        # compute the baseline signal
        baseline = self.__call__(scheme)

        forward_diff = self._simulate_signals(self._parameter_vectors_forward, scheme)
        backward_diff = self._simulate_signals(self._parameter_vectors_backward, scheme)

        central_diff = forward_diff - backward_diff

        if np.all(central_diff == 0):
            raise RuntimeError("Central differences evaluate to 0, probably error in the way signals are simulated?")

        # reset parameters to original
        self.set_parameters_from_vector(self._parameter_baseline)

        # return jacobian
        jac = np.concatenate((central_diff * self._reciprocal_h, [baseline])).T
        return jac[:, self.include_optimize]

    def _simulate_signals(self, parameter_vectors: np.ndarray, scheme: AcquisitionScheme) -> np.ndarray:
        """

        :param parameter_vectors:
        :param scheme:
        :return:
        """
        # number of parameter vectors
        npv = parameter_vectors.shape[0]
        signals = np.zeros((npv, scheme.pulse_count))
        for i in range(npv):
            self.set_parameters_from_vector(parameter_vectors[i, :])
            signals[i, :] = self.__call__(scheme)

        # self.set_parameters_from_vector(self._parameter_baseline)
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

    def scaled_jacobian(self, scheme: AcquisitionScheme):
        # Extracting the jacobian w.r.t the included parameters only
        # casting to numpy array if not done already
        include = self.include_optimize
        scales = self.scales
        jac = self.jacobian(scheme)

        N_measurements = scheme.pulse_count

        # Scaling the jacobian
        jac_rescaled = jac * scales[include]
        return jac_rescaled

    @property
    def parameters(self) -> Dict[str, Union[float, np.ndarray]]:
        parameters = {}
        for name, parameter in self.items():
            parameters[name] = parameter.value
        return parameters

    @property
    def parameter_names(self) -> List[str]:
        return [key for key in self.keys()]

    def set_parameters_from_vector(self, new_parameter_values: np.ndarray) -> None:
        """
        You should probably overwrite this method if you are using a wrapped model.

        :param new_parameter_values:
        :return:
        """
        for parameter, new_value in zip(self.values(), new_parameter_values):
            parameter.value = new_value

    def set_fit_parameters(self, new_values: Union[np.ndarray, dict]) -> None:
        """
        Sets the TissueParameters that are flagged for fitting to the provided values.

        :param new_values: The parameter values to be set in vector form or dictionary form
        :return: None
        """
        if isinstance(new_values, dict):
            # check length
            val_lst = list(new_values.values())
            # Reenter the function as an array
            self.set_fit_parameters(np.array(val_lst))
        else:
            if np.sum(self.include_fit) != new_values.shape[0]:
                raise ValueError("Shape of new values does not match number of parameters marked for fitting.")

            for i, parameter in enumerate(self.get_fit_parameters()):
                if parameter.fit_flag:
                    parameter.value = new_values[i]

    def get_fit_parameters(self):
        return np.array(list(self.values()))[self.include_fit]

    @property
    def parameter_vector(self) -> np.ndarray:
        # put all parameter values in a single array
        vector = np.array([parameter.value for parameter in self.values()])
        return vector

    @property
    def scales(self):
        return np.array([value.scale for value in self.values()])

    @property
    def include_optimize(self):
        return np.array([value.optimize for value in self.values()])

    @property
    def include_fit(self):
        return np.array([value.fit_flag for value in self.values()])

    @property
    def fit_bounds_all(self) -> Bounds:
        lbs = []
        ubs = []
        for parameter in self.get_fit_parameters():
            ub = parameter.fit_bounds[1]
            lb = parameter.fit_bounds[0]
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
    def fit_initial_guess(self) -> np.ndarray:
        return np.array([self[key].fit_guess for key in np.array(self.parameter_names)[self.include_fit]])

    def print_comparison(self, other: TissueModel):
        """
        Method for comparing two tissue models of the same type
        :param other:
        :return:
        """
        for k_, v_me, v_him in zip(self.parameter_names, self.parameter_vector, other.parameter_vector):
            print(k_, v_me, v_him)


class MultiTissueModel(TissueModel):
    def __init__(self, models: List[TissueModel], volume_fractions: List[float]):

        self._models = models
        self.N_models = len(models)
        # making a parameter dictionary using parameters in the individual compartments
        parameters = {}
        for i, model in enumerate(models):
            for key, value in model.items():
                if key != BASE_SIGNAL_KEY:
                    parameters.update({MODEL_PREFIX + f"{i}_" + key: value})

        # Inserting the partial volumes as model parameters
        if len(volume_fractions) != self.N_models:
            raise ValueError("Not enough volume fractions provided for number of models")
        if sum(volume_fractions) != 1.:
            raise ValueError("Volume fractions dont sum to 1")

        for i, vf in enumerate(volume_fractions):
            parameters.update({
                VOLUME_FRACTION_PREFIX + f"{i}": TissueParameter(value=vf, scale=1., fit_bounds=(0.0, 1.0))
            })

        # Add S0 as a tissue parameter (to be excluded in parameters extraction etc.)
        parameters.update({BASE_SIGNAL_KEY: TissueParameter(value=1.0, scale=1.0, optimize=False, fit_flag=False,
                                                            fit_bounds=(0.0, 2.0))})
        super().__init__(parameters)

    def set_parameters_from_vector(self, new_parameter_values: np.ndarray) -> None:
        # Make sure that the parameters are updated on the individual models first
        i = 0
        for model in self._models:
            N_p = len(model)
            model.set_parameters_from_vector(new_parameter_values[i:N_p])
            i += N_p

        super().set_parameters_from_vector(new_parameter_values)

    def set_fit_parameters(self, new_values: Union[np.ndarray, dict]) -> None:
        super().set_fit_parameters(new_values)

        if isinstance(new_values, dict):
            new_values = np.array(list(new_values.values()))
        # We also update the parameters on the models in this object
        i = 0
        for model in self._models:
            N_flagged = np.sum(model.include_fit)
            model.set_fit_parameters(new_values[i:(i + N_flagged)])
            i += N_flagged

    def __call__(self, scheme: AcquisitionScheme) -> np.ndarray:
        # use the call functions of the models
        compartment_signals = np.stack([model(scheme) for model in self._models], axis=-1)
        return np.sum(compartment_signals * self.volume_fractions, axis=-1)

    def fit(self, scheme: AcquisitionScheme, signal: np.ndarray, method: Union[str, callable] = 'trust-constr',
            **fit_options) -> FittedModelMinimize:

        cost_fun_args = (signal, scheme, deepcopy(self))

        # scaling the bounds
        bounds = self.fit_bounds_all
        bounds.ub /= self.scales[self.include_fit]
        bounds.lb /= self.scales[self.include_fit]

        x0 = self.fit_initial_guess / self.scales[self.include_fit]

        if method == 'DE':
            result = differential_evolution(fit_cost,
                                            args=cost_fun_args,
                                            bounds=bounds,
                                            constraints=self.fit_constraints,
                                            workers=-1,
                                            updating='deferred',
                                            disp=True)
        else:

            rng = default_rng()
            machine_epsilon = np.finfo(float).eps
            N_INIT = 10
            best_cost = np.inf
            best_result = None
            for _ in range(N_INIT):
                # Generate initial guess in bounds where we account for machine precision to prevent stepping out

                x0 = rng.uniform(low=bounds.lb + machine_epsilon, high=bounds.ub - machine_epsilon, size=bounds.lb.size)

                result = minimize(fit_cost, x0=x0, args=cost_fun_args,
                                  bounds=bounds,
                                  constraints=self.fit_constraints,
                                  method=method, options={})

                if result.fun < best_cost:
                    best_cost = result.fun
                    best_result = result

            result = best_result

        result.x = result.x * self.scales[self.include_fit]
        return FittedModelMinimize(self, result)

    @property
    def fit_constraints(self) -> Constraints:
        if self.N_models == 1:
            return ()

        # for now only volume fractions.
        A = []
        for name in np.array(self.parameter_names)[self.include_fit]:
            if name.startswith(VOLUME_FRACTION_PREFIX):
                A.append(1)
            else:
                A.append(0)
        return LinearConstraint(A, 1, 1, keep_feasible=False)

    @property
    def volume_fractions(self) -> np.ndarray:
        vfs = []
        for key, parameter in self.items():
            if key.startswith(VOLUME_FRACTION_PREFIX):
                vfs.append(parameter.value)
        return np.array(vfs)


class RelaxedMultiTissueModel(MultiTissueModel):
    def __init__(self, models: List[TissueModel], volume_fractions: List[float],
                 relaxation_times: Union[List[float], np.ndarray]):
        super().__init__(models, volume_fractions)
        # pop the base signal
        base_signal = self.pop(BASE_SIGNAL_KEY)
        # insert relaxation times
        insert_relaxation_times(relaxation_times, self, self.N_models)
        # reinsert the base signal
        self.update({BASE_SIGNAL_KEY: base_signal})

    @property
    def relaxation_times(self) -> np.ndarray:
        rts = []

        for key, value in self.items():
            if key.startswith(RELAXATION_PREFIX):
                rts.append(value.value)
        return np.array(rts)

    def __call__(self, scheme: AcquisitionScheme) -> np.ndarray:
        # use the call functions of the models
        compartment_signals = np.stack([model(scheme) for model in self._models], axis=-1)
        echo_times = scheme["EchoTime"].values
        relaxation_decay = np.exp(- echo_times[:, np.newaxis] / self.relaxation_times)
        return np.sum(compartment_signals * self.volume_fractions * relaxation_decay, axis=-1)


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
                "Minimize says optimization was unsuccesfull inspect fit information to decide on further actions.",
                category=RuntimeWarning)

    def print_fit_information(self) -> None:
        for key in self.result.keys():
            print(key, self.result[key])

    @property
    def fitted_parameters(self) -> Dict[str, np.ndarray]:
        fit_values = self.result.x
        out = {}
        for i, name in enumerate(np.array(self.model.parameter_names)[self.model.include_fit]):
            out.update({name: fit_values[i]})
        return out


# TODO: Take T2* and relaxation parameter distributions into account. See eq. 5 and 6 in
#  https://www.ncbi.nlm.nih.gov/books/NBK567564/
class RelaxationTissueModel(TissueModel):
    """
    Defines a tissue by its relaxation parameters.

    :param t1: Longitudinal relaxation time constant T1 in milliseconds.
    :param t2: Transverse relaxation time constant T2 in milliseconds.
    :param s0: MR signal from fully recovered magnetisation, just before the 90° RF pulse.
    """

    def __init__(self, t1: float, t2: float, s0: float = 1.0):
        super().__init__({
            T1_KEY: TissueParameter(value=t1, scale=t1, optimize=False),
            T2_KEY: TissueParameter(value=t2, scale=t2),
            BASE_SIGNAL_KEY: TissueParameter(value=s0, scale=s0, optimize=False),
        })

    # TODO: Support other relaxation-acquisition schemes, e.g. Union[SpinEchoAcquisitionScheme,
    #  InversionRecoveryAcquisitionScheme] and switch model based on these.
    def __call__(self, scheme: InversionRecoveryAcquisitionScheme) -> np.ndarray:
        ti = scheme.inversion_times  # ms
        tr = scheme.repetition_times  # ms
        te = scheme.echo_times  # ms

        ti_t1 = np.exp(-ti / self[T1_KEY].value)
        tr_t1 = np.exp(-tr / self[T1_KEY].value)
        te_t2 = np.exp(-te / self[T2_KEY].value)

        # Rather than varying TR to achieve different T1 weightings, Mulkern et al. (2000) incorporate an inversion
        # pulse prior to the 90° pulse in the diffusion-weighted SE sequence for simultaneous D-T1 measurement.
        #
        # See section 7.4.2 of 'Advanced Diffusion Encoding Methods in MRI', Topgaard D, editor (2020):
        # https://www.ncbi.nlm.nih.gov/books/NBK567564
        return self[BASE_SIGNAL_KEY].value * (1 - 2 * ti_t1 + tr_t1) * te_t2

    def jacobian(self, scheme: InversionRecoveryAcquisitionScheme) -> np.ndarray:
        ti = scheme.inversion_times  # ms
        tr = scheme.repetition_times  # ms
        te = scheme.echo_times  # ms

        ti_t1 = np.exp(-ti / self[T1_KEY].value)
        tr_t1 = np.exp(-tr / self[T1_KEY].value)
        te_t2 = np.exp(-te / self[T2_KEY].value)

        # Calculate the derivative of the signal attenuation to T1, T2 and S0.
        jac = np.array([
            self[BASE_SIGNAL_KEY].value * (-2 * ti * ti_t1 + tr * tr_t1) * te_t2 / (self[T1_KEY].value ** 2),
            # δS(S0, T1, T2) / δT1
            self[BASE_SIGNAL_KEY].value * te * (1 - 2 * ti_t1 + tr_t1) * te_t2 / (self[T2_KEY].value ** 2),
            # δS(S0, T1, T2) / δT2
            (1 - 2 * ti_t1 + tr_t1) * te_t2,  # δS(S0, T1, T2) / δS0
        ]).T
        return jac[:, self.include_optimize]

    def fit(self, scheme: InversionRecoveryAcquisitionScheme, signal: np.ndarray,
            **fit_options) -> FittedModelCurveFit:
        ti = scheme.inversion_times  # ms
        tr = scheme.repetition_times  # ms
        te = scheme.echo_times  # ms

        # whether parameters are included in the fit
        include = self.include_optimize

        # Using the current model parameters as initials for the fit (these are the *true* values!)
        initial_parameters = np.array([param.value for param in self.values()])

        # The signal function we fit to extract the tissue parameters
        def signal_fun(measurement, t1, t2, s0):
            if not include[0]:
                t1 = initial_parameters[0]
            if not include[1]:
                t2 = initial_parameters[1]
            if not include[2]:
                s0 = initial_parameters[2]

            ti_t1 = np.exp(-ti / t1)
            tr_t1 = np.exp(-tr / t1)
            te_t2 = np.exp(-te / t2)
            return s0 * (1 - 2 * ti_t1 + tr_t1) * te_t2

        # Tissue induced bounds on the parameters ( T1 < 7000 , T2 < 3000 )
        # TODO: add bounds as a tissueparameter attribute

        bounds = (np.array([0, 0, 0]), np.array([7000, 3000, np.inf]))

        # The scipy fitting routine
        result = curve_fit(signal_fun, np.arange(len(tr)), signal, initial_parameters, bounds=bounds,
                           full_output=True, **fit_options)

        return FittedModelCurveFit(self, result)


class ExponentialTissueModel(TissueModel):
    def __init__(self, T2: float, S0: float = 1.0):
        """

        :param T2: The T2 relaxation in [ms]
        :param S0: The initial signal
        """
        super().__init__({
            T2_KEY: TissueParameter(value=T2, scale=1.0, optimize=True, fit_bounds=(.1, 10e3)),
            BASE_SIGNAL_KEY: TissueParameter(value=S0, scale=1.0, optimize=True, fit_bounds=(.1, 2))
        })

    def __call__(self, scheme: EchoScheme) -> np.ndarray:
        """
        Implement the signal equation S = S0 * exp(-TE/T2) here
        :return:
        """
        TE = scheme.echo_times
        T2 = self[T2_KEY].value
        S0 = self[BASE_SIGNAL_KEY].value
        return S0 * np.exp(- TE / T2)

    def jacobian(self, scheme: EchoScheme) -> np.ndarray:
        """
        This is the analytical way of computing the jacobian.
        :param scheme:
        :return:
        """
        TE = scheme.echo_times
        T2 = self[T2_KEY].value
        S0 = self[BASE_SIGNAL_KEY].value

        # the base signal
        S = S0 * np.exp(-TE / T2)
        # return np.array([-TE * S, 1]).T
        jac = np.array([(TE / T2 ** 2) * S, S / S0]).T
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
        def signal_fun(measurement, t2, s0):
            if not include[0]:
                t2 = parameter_value[0]
            if not include[1]:
                s0 = parameter_value[1]

            return s0 * np.exp(-te / t2)

        initial_guess = self.scales
        result = curve_fit(signal_fun, np.arange(len(te)), signal, initial_guess,
                           bounds=(self.fit_bounds_all.lb, self.fit_bounds_all.ub),
                           maxfev=4 ** 2 * 100, full_output=True, **fit_options)

        return FittedModelCurveFit(self, result)


class RelaxedIsotropicModel(TissueModel):
    def fit(self, scheme: AcquisitionScheme, signal: np.ndarray, **fit_options) -> FittedModelCurveFit:
        raise NotImplementedError()

    def __init__(self, t2: float, diffusivity: float, s0: float = 1.0):
        """
        :param t2: The tissues T2 in [ms]
        :param diffusivity: The tissue diffusivity in [mm^2 / s]
        :param s0: signal at the zeroth measurement [dimensionless]
        """
        super().__init__({
            T2_KEY: TissueParameter(value=t2, scale=t2),
            DIFFUSIVITY_KEY: TissueParameter(value=diffusivity, scale=diffusivity),
            BASE_SIGNAL_KEY: TissueParameter(value=s0, scale=s0, optimize=False),
        })

    def __call__(self, scheme: ReducedDiffusionScheme) -> np.ndarray:
        # The signal equation for this model S = S0*exp(-b.*D).*exp(-TE/T2)

        bvalues = scheme.b_values
        te = scheme.echo_times

        b_D = np.exp(-bvalues * self[DIFFUSIVITY_KEY].value)
        te_t2 = np.exp(- te / self[T2_KEY].value)
        return self[BASE_SIGNAL_KEY].value * b_D * te_t2

    def jacobian(self, scheme: ReducedDiffusionScheme) -> np.ndarray:
        # Acquisition parameters
        bvalues = scheme.b_values
        te = scheme.echo_times

        # tissuemodel parameters
        D = self[DIFFUSIVITY_KEY].value
        T2 = self[T2_KEY].value
        S0 = self[BASE_SIGNAL_KEY].value

        # Exponents
        b_D = np.exp(-bvalues * D)
        te_t2 = np.exp(- te / T2)

        jac = np.array([te * S0 * b_D * te_t2 / T2 ** 2, - bvalues * S0 * b_D * te_t2, b_D * te_t2]).T
        return jac[:, self.include_optimize]


class TissueModelDecorator(TissueModel, ABC):
    """
    Abstract class for initialization of TissueModel decorators. this just passes all public methods to the original object
    override these methods to extend or alter functionality while retaining the same interface.

    This concept is based on the decorator design pattern and more information can be found at
    https://refactoring.guru/design-patterns/decorator
    """

    def __init__(self, original: TissueModel):
        self._original = deepcopy(original)

    def __call__(self, scheme: AcquisitionScheme) -> np.ndarray:
        return self._original.__call__(scheme)

    def __str__(self):
        return self._original.__str__()

    def jacobian(self, scheme: AcquisitionScheme) -> np.ndarray:
        return self._original.jacobian(scheme)

    def fit(self, scheme: AcquisitionScheme, signal: np.ndarray, **fit_options) -> FittedModel:
        return self._original.fit(scheme, signal, **fit_options)

    @property
    def parameters(self):
        return self._original.parameters

    @property
    def parameter_names(self):
        return self._original.parameter_names

    @property
    def scales(self):
        return self._original.scales

    @property
    def include_optimize(self):
        return self._original.include_optimize

    @property
    def parameter_vector(self) -> np.ndarray:
        return self._original.parameter_vector


def insert_relaxation_times(relaxation_times, tissue_model, N_models):
    """
    This function inserts relaxation times into the parameters dictionary

    :param relaxation_times:
    :param tissue_model:
    :param N_models:
    :return:
    """
    # Add relaxation times (if none are provided we set them to inifinity and exclude from optimization
    relax_opt_flag = True
    relax_fit_flag = True
    if relaxation_times is None:
        relax_opt_flag = False
        relax_fit_flag = False
        relaxation_times = [np.inf for _ in range(N_models)]

    # converting T2 to array
    if not isinstance(relaxation_times, np.ndarray):
        relaxation_times = np.array(relaxation_times, dtype=float)

    # check the number of relaxivities with the number of models
    if relaxation_times.size != N_models:
        raise ValueError("Specifiy relaxation for all compartments")

    # store relaxations as tissue_parameters
    if relaxation_times.size > 1:
        for i, value in enumerate(relaxation_times):
            tissue_model.update({RELAXATION_PREFIX + str(i):
                                     TissueParameter(value, 1.0,
                                                     optimize=relax_opt_flag,
                                                     fit_flag=relax_fit_flag,
                                                     fit_bounds=RELAXATION_BOUNDS)})
    else:
        tissue_model.update(
            {RELAXATION_PREFIX + "0": TissueParameter(float(relaxation_times), 1.0, optimize=relax_opt_flag,
                                                      fit_flag=relax_fit_flag,
                                                      fit_bounds=RELAXATION_BOUNDS)})


def fit_cost(fit_parameter_vector, signal, scheme, model: TissueModel):
    model.set_fit_parameters(fit_parameter_vector * model.scales[model.include_fit])
    predicted_signal = model(scheme)
    square_diff = (signal - predicted_signal) ** 2
    return np.sum(square_diff)


def fit_residuals(fit_parameter_vector, signal, scheme, model: TissueModel):
    model.set_fit_parameters(fit_parameter_vector)
    predicted_signal = model(scheme)
    return signal - predicted_signal

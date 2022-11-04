"""
The TissueModel is a base class for different kinds of tissue models (different types of models and/or different types
 of modelling-software).

In order to simulate the MR signal in response to a MICROtool acquisition scheme, the TissueModel first translates the
 acquisition scheme to a modelling-software specific one.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Union, List

import numpy as np
from scipy.optimize import curve_fit
from tabulate import tabulate

from .acquisition_scheme import AcquisitionScheme, InversionRecoveryAcquisitionScheme, EchoScheme, \
    FlaviusAcquisitionScheme


@dataclass
class TissueParameter:
    # noinspection PyUnresolvedReferences
    """
    Defines a scalar tissue parameter and its properties.

    :param value: Parameter value.
    :param scale: The typical parameter value scale (order of magnitude).
    :param optimize: Specifies if the parameter should be included in the optimization (default = True).
    """
    value: float
    scale: float
    optimize: bool = True

    def __str__(self):
        return f'{self.value} (scale: {self.scale}, optimize: {self.optimize})'


class TissueModel(Dict[str, TissueParameter], ABC):
    @abstractmethod
    def __call__(self, scheme: AcquisitionScheme) -> np.ndarray:
        """
        Calculates the MR signal attenuation.

        :param scheme: An AcquisitionScheme.
        :return: A array with N signal attenuation values, where N is the number of samples.
        """
        pass

    def __str__(self) -> str:

        table = []
        for key, value in self.items():
            table.append([key, value.value, value.scale, value.optimize])

        table_str = tabulate(table, headers=["Tissueparameter", "Value", "Scale", "Optimize"])

        return f'Tissue model with {len(self)} scalar parameters:\n{table_str}'

    @abstractmethod
    def jacobian(self, scheme: AcquisitionScheme) -> np.ndarray:
        """
        Calculates the change in MR signal attenuation due to a change in the tissue model parameters.

        The order of the tissue parameters is the same as returned by get_parameters().

        :param scheme: An AcquisitionScheme.
        :return: An N×M Jacobian matrix, where N is the number of samples and M is the number of tissue parameters.
        """
        pass

    @abstractmethod
    def fit(self, scheme: AcquisitionScheme, signal: np.ndarray, **fit_options) -> FittedTissueModel:
        """
        Fits the tissue model parameters to noisy_signal data given an acquisition scheme.
        :param signal: The noisy signal
        :param scheme: The scheme under investigation
        :return: A tuple containing the optimized tissue parameters as a TissueModel instance and the covariance matrix
                 of the fit
        """

        pass

    @property
    def parameters(self) -> Dict[str, Union[float, np.ndarray]]:
        parameters = {}
        for name, parameter in self.items():
            parameters[name] = parameter.value
        return parameters

    @property
    def parameter_names(self) -> List[str]:
        return [key for key in self.keys()]

    @property
    def scales(self):
        return [value.scale for value in self.values()]

    @property
    def include(self):
        return [value.optimize for value in self.values()]


class FittedTissueModel:
    def __init__(self, model: TissueModel, fitted_parameters_vector: np.ndarray):
        self._model = model
        self.fitted_parameters_vector = fitted_parameters_vector

    @property
    def fitted_parameters(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError("Still need to correct this.")
        # # getting the parameter names and values
        # names = self._model.parameter_names
        # parameter_vector = self.fitted_parameters_vector
        # out = {}
        # for i,name in enumerate(names):
        #     if self._model[name]:
        #         pass
        # # TODO: deal with fixed parameters
        # return {name: parameter_vector[i] for i, name in enumerate(names)}


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
            'T1': TissueParameter(value=t1, scale=t1, optimize=False),
            'T2': TissueParameter(value=t2, scale=t2),
            'S0': TissueParameter(value=s0, scale=s0, optimize=False),
        })

    # TODO: Support other relaxation-acquisition schemes, e.g. Union[SpinEchoAcquisitionScheme,
    #  InversionRecoveryAcquisitionScheme] and switch model based on these.
    def __call__(self, scheme: InversionRecoveryAcquisitionScheme) -> np.ndarray:
        ti = scheme.inversion_times  # ms
        tr = scheme.repetition_times  # ms
        te = scheme.echo_times  # ms

        ti_t1 = np.exp(-ti / self['T1'].value)
        tr_t1 = np.exp(-tr / self['T1'].value)
        te_t2 = np.exp(-te / self['T2'].value)

        # Rather than varying TR to achieve different T1 weightings, Mulkern et al. (2000) incorporate an inversion
        # pulse prior to the 90° pulse in the diffusion-weighted SE sequence for simultaneous D-T1 measurement.
        #
        # See section 7.4.2 of 'Advanced Diffusion Encoding Methods in MRI', Topgaard D, editor (2020):
        # https://www.ncbi.nlm.nih.gov/books/NBK567564
        return self['S0'].value * (1 - 2 * ti_t1 + tr_t1) * te_t2

    def jacobian(self, scheme: InversionRecoveryAcquisitionScheme) -> np.ndarray:
        ti = scheme.inversion_times  # ms
        tr = scheme.repetition_times  # ms
        te = scheme.echo_times  # ms

        ti_t1 = np.exp(-ti / self['T1'].value)
        tr_t1 = np.exp(-tr / self['T1'].value)
        te_t2 = np.exp(-te / self['T2'].value)

        # Calculate the derivative of the signal attenuation to T1, T2 and S0.
        jac = np.array([
            self['S0'].value * (-2 * ti * ti_t1 + tr * tr_t1) * te_t2 / (self['T1'].value ** 2),  # δS(S0, T1, T2) / δT1
            self['S0'].value * te * (1 - 2 * ti_t1 + tr_t1) * te_t2 / (self['T2'].value ** 2),  # δS(S0, T1, T2) / δT2
            (1 - 2 * ti_t1 + tr_t1) * te_t2,  # δS(S0, T1, T2) / δS0
        ]).T
        return jac

    def fit(self, scheme: InversionRecoveryAcquisitionScheme, signal: np.ndarray,
            **fit_options) -> FittedTissueModel:
        ti = scheme.inversion_times  # ms
        tr = scheme.repetition_times  # ms
        te = scheme.echo_times  # ms

        # whether or not parameters are included in the fit
        include = np.array([param.optimize for param in self.values()])

        # Using the current model parameters as initials for the fit (these are the *true* values!)
        initial_parameters = np.array([param.value for param in self.values()])

        # The signal function we fit to extract the tissueparameters
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
        popt, _ = curve_fit(signal_fun, np.arange(len(tr)), signal, initial_parameters, bounds=bounds)

        return FittedTissueModel(self, popt)


class ExponentialTissueModel(TissueModel):
    def __init__(self, T2: float, S0: float = 1.0):
        """
        Set the tissue parameters
        """
        super().__init__({
            'T2': TissueParameter(value=T2, scale=T2, optimize=True),
            'S0': TissueParameter(value=S0, scale=S0, optimize=True)
        })

    def __call__(self, scheme: EchoScheme) -> np.ndarray:
        """
        Implement the signal equation S = S0 * exp(-TE/T2) here
        :return:
        """
        TE = scheme.echo_times
        T2 = self['T2'].value
        S0 = self['S0'].value
        return S0 * np.exp(- TE / T2)

    def jacobian(self, scheme: EchoScheme) -> np.ndarray:
        """
        This is the analytical way of computing the jacobian.
        :param scheme:
        :return:
        """
        TE = scheme.echo_times
        T2 = self['T2'].value
        S0 = self['S0'].value

        # the base signal
        S = S0 * np.exp(-TE / T2)
        # return np.array([-TE * S, 1]).T
        return np.array([(TE / T2 ** 2) * S, S / S0]).T

    def jacobian_num(self, scheme: EchoScheme) -> np.ndarray:
        """
        Uses finite differences to compute the
        :param scheme:
        :return:
        """
        raise NotImplementedError()

    def fit(self, scheme: EchoScheme, signal: np.ndarray, **fit_options) -> FittedTissueModel:
        """

        :param scheme:
        :param signal:
        :param fit_options:
        :return:
        """
        # extracting the echo times from the scheme
        te = scheme.echo_times

        # Whether or not to include a parameter in the fitting process?
        include = self.include

        # initial parameters in case we want to exclude some parameter from the fitting process
        initial_parameters = list(self.parameters.values())

        # the function defining the signal in form compatible with scipy curve fitting
        def signal_fun(measurement, t2, s0):
            if not include[0]:
                t2 = initial_parameters[0]
            if not include[1]:
                s0 = initial_parameters[1]

            return s0 * np.exp(-te / t2)

        # TODO create default value and add as a fitting option
        # hard coding the fitting bounds for now
        bounds = (np.array([0, 0]), np.array([np.inf, np.inf]))
        popt, _ = curve_fit(signal_fun, np.arange(len(te)), signal, initial_parameters, bounds=bounds,
                            maxfev=4 ** 2 * 100)

        return FittedTissueModel(self, popt)


class FlaviusSignalModel(TissueModel):
    def __init__(self, t2: float, diffusivity: float, s0: float = 1.0):
        """
        :param t2: The tissues T2 in [ms]
        :param diffusivity: The tissue diffusivity in [mm^2 / s]
        :param s0: signal at the zeroth measurement [dimensionless]
        """
        super().__init__({
            'T2': TissueParameter(value=t2, scale=t2),
            'Diffusivity': TissueParameter(value=diffusivity, scale=diffusivity),
            'S0': TissueParameter(value=s0, scale=s0, optimize=False),
        })

    def __call__(self, scheme: FlaviusAcquisitionScheme) -> np.ndarray:
        # The signal equation for this model S = S0*exp(-b.*D).*exp(-TE/T2)

        bvalues = scheme.b_values
        te = scheme.echo_times

        b_D = np.exp(-bvalues * self['Diffusivity'].value)
        te_t2 = np.exp(- te / self['T2'].value)
        return self['S0'].value * b_D * te_t2

    def jacobian(self, scheme: FlaviusAcquisitionScheme) -> np.ndarray:
        # Acquisition parameters
        bvalues = scheme.b_values
        te = scheme.echo_times

        # tissuemodel parameters
        D = self['Diffusivity'].value
        T2 = self['T2'].value
        S0 = self['S0'].value

        # Exponents
        b_D = np.exp(-bvalues * D)
        te_t2 = np.exp(- te / T2)

        jac = [te * S0 * b_D * te_t2 / T2 ** 2, - bvalues * S0 * b_D * te_t2, b_D * te_t2]
        return np.array(jac).T


class TissueModelDecoratorBase(TissueModel, ABC):
    """
    Abstract class for initialization of TissueModel decorators. this just passes all public methods to the original object
    override these methods to extend or alter functionality while retaining the same interface.
    """

    def __init__(self, original: TissueModel):
        self._original = original
        super().__init__(original)

    def __call__(self, scheme: AcquisitionScheme) -> np.ndarray:
        return self._original.__call__(scheme)

    def __str__(self):
        return self._original.__str__()

    def jacobian(self, scheme: AcquisitionScheme) -> np.ndarray:
        return self._original.jacobian(scheme)

    def fit(self, scheme: AcquisitionScheme, signal: np.ndarray, **fit_options) -> FittedTissueModel:
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
    def include(self):
        return self._original.include

from __future__ import annotations

from copy import copy, deepcopy
from typing import Dict, List, Optional, Union

import numpy as np
from dmipy.core.acquisition_scheme import DmipyAcquisitionScheme, acquisition_scheme_from_bvalues
from dmipy.core.fitted_modeling_framework import FittedMultiCompartmentModel
from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.signal_models.gaussian_models import G1Ball

from microtool.acquisition_scheme import DiffusionAcquisitionScheme
from microtool.scanner_parameters import ScannerParameters, default_scanner
from microtool.tissue_model import TissueModel, TissueParameter, TissueModelDecoratorBase, FittedModel


# TODO: deal with fractional parameter relations!
def get_parameters(multi_model: MultiCompartmentModel) -> Dict[str, TissueParameter]:
    """
    Compiles a dictionary of all tissue parameters present in the given multi-compartment model.
    These are wrapped in the microtool TissueParameter wrapper. Parameter linking and optimization flags are extracted.

    :param multi_model: A dmipy multi-compartment model.
    :return: A dictionary of tissueparameters
    """

    # Iterate over all tissue models in the MultiCompartmentModel.
    parameters = {}
    for model, model_name in zip(multi_model.models, multi_model.model_names):
        # Iterate over all scalar and vector parameters in the tissue model.
        for parameter_name in model.parameter_names:
            value = np.array(getattr(model, parameter_name), dtype=np.float64, copy=True)
            scale = model.parameter_scales[parameter_name]
            cardinality = model.parameter_cardinality[parameter_name]

            # Determine if the parameter is fixed in the multicompartment model before wrapping
            linked_or_fixed = False

            # dmipy removes any fixed or linked parameters from MultiCompartentModel names.
            if model_name + parameter_name not in multi_model.parameter_names:
                linked_or_fixed = True

            if value is None:
                raise ValueError(f'Parameter {parameter_name} of model {model_name} has no value.')
            if np.any(np.isnan(value)):
                raise ValueError(f'Parameter {parameter_name} of model {model_name} has a nan value.')
            # Iterate over vector parameters and add their elements as scalar tissue parameters.
            for i in range(cardinality):
                # Determine whether the tissue parameter was fixed before wrapping
                index_postfix = '' if cardinality == 1 else f'_{i}'
                parameters[model_name + parameter_name + index_postfix] = TissueParameter(
                    value=value[i] if cardinality > 1 else value,
                    scale=scale[i] if cardinality > 1 else scale,
                    # non fixed or linked parameters can be included in the optimization.
                    optimize=not linked_or_fixed
                )

    return parameters


def convert_diffusion_scheme2dmipy_scheme(scheme: DiffusionAcquisitionScheme) -> DmipyAcquisitionScheme:
    """
    Takes in a scheme from the microtool toolbox and returns a scheme for the dmipy toolbox. It should be noted that the
    dmipy toolbox takes SI unit acquisition parameters only. Therefore we convert the microtool parameters.
    Additionally, this conversion loses the echo time information!
    This is due to the fact that dmipy has the notion that echo times are constrained to the same shell.
    We use a more relaxed approach.

    :param scheme: DiffusionAcquisitionScheme
    :return: DmipyAcquisitionScheme
    """
    if not isinstance(scheme, DiffusionAcquisitionScheme):
        raise TypeError(f"scheme is of type {type(scheme)}, we expected an {DiffusionAcquisitionScheme}")
    # note that dmipy has a different notion of echo times so they are not included in the conversion
    return acquisition_scheme_from_bvalues(
        scheme.b_values * 1e6,  # Convert from s/mm² to s/m².
        scheme.b_vectors,
        scheme.pulse_widths * 1e-3,  # Convert from ms to s.
        scheme.pulse_intervals * 1e-3,  # Convert from ms to s.
    )


def convert_dmipy_scheme2diffusion_scheme(scheme: DmipyAcquisitionScheme,
                                          scanner_parameters: ScannerParameters = default_scanner) \
        -> DiffusionAcquisitionScheme:
    """
    Takes a scheme from the dmipy toolbox and converts to scheme from microtool toolbox. We convert units since
    microtool has non SI units.

    :param scheme: DmipyAcquisitionScheme
    :param scanner_parameters: Scanner specific parameters
    :return: DiffusionAcquisitionScheme
    """
    if not isinstance(scheme, DmipyAcquisitionScheme):
        raise TypeError(f"scheme is of type {type(scheme)}, we expected an {DmipyAcquisitionScheme}")

    # convert to s/mm^2 from s/m^2
    b_values = scheme.bvalues * 1e-6
    b_vectors = scheme.gradient_directions
    # convert to ms from s
    pulse_widths = scheme.delta * 1e3
    pulse_intervals = scheme.Delta * 1e3
    echo_times_SI = scheme.TE
    if echo_times_SI is None:
        return DiffusionAcquisitionScheme(b_values, b_vectors, pulse_widths, pulse_intervals,
                                          scan_parameters=scanner_parameters)
    else:
        return DiffusionAcquisitionScheme(b_values, b_vectors, pulse_widths, pulse_intervals, echo_times_SI * 1e3,
                                          scan_parameters=scanner_parameters)


class DmipyTissueModel(TissueModel):
    """
    Wrapper for the MultiCompartment models used by dmipy. Note that the parameters need to be initialized in the
    dmipy model otherwise a value error is raised.
    """
    _model: MultiCompartmentModel  # Reminder that we store the dmipy multicompartment model in this attribute

    def __init__(self, model: MultiCompartmentModel, volume_fractions: List[float] = None):
        """

        :param model: MultiCompartment model
        :param volume_fractions: The relative volume fractions of the models (order in the same way you initialized the
                                 multicompartment model)
        """
        super().__init__()
        # Extract the tissue parameters from individual models and convert to 'scalars'.
        self.update(get_parameters(model))

        # -------------------- Set up volume fractions if there are multiple models
        if model.N_models > 1:
            if volume_fractions is None:
                raise ValueError("Provide volume fractions if composite tissuemodels are used.")
            # Get the ordered partial volume names from the model
            vf_keys = model.partial_volume_names
            # check if the volume_fractions match the length of this dictionary
            if len(volume_fractions) != len(vf_keys):
                raise ValueError("Not enough volume fractions provided for the number of models.")
            if np.array(volume_fractions).sum() != 1:
                raise ValueError("Provide volume fractions that sum to 1.")
            # Including the volume fractions as TissueParameters to the DmipyTissueModel
            for i, key in enumerate(vf_keys):
                self.update({key: TissueParameter(value=volume_fractions[i], scale=1.)})

        # Add S0 as a tissue parameter (to be excluded in parameters extraction etc.)
        self.update({'S0': TissueParameter(value=1.0, scale=1.0, optimize=False)})
        self._model = model

        # ----------------------------JACOBIAN HELPER VARIABLES (finite differences)

        # Get the baseline parameter vector, but don't include S0.
        self._parameter_baseline = np.array([parameter.value for parameter in self.values()])[:-1]

        # Calculate finite differences and corresponding parameter vectors for calculating derivatives.
        step_size = 1e-6
        h = np.array([parameter.scale * step_size for parameter in self.values()])[:-1]
        self._parameter_vectors_forward = self._parameter_baseline + 0.5 * np.diag(h)
        self._parameter_vectors_backward = self._parameter_baseline - 0.5 * np.diag(h)
        self._reciprocal_h = (1 / h).reshape(-1, 1)

    def __call__(self, scheme: DiffusionAcquisitionScheme) -> np.ndarray:
        dmipy_scheme = convert_diffusion_scheme2dmipy_scheme(scheme)

        # Evaluate the dmipy model.
        s0 = self['S0'].value

        # dont include s0 for dmipy simulate signal
        parameters = self.parameter_vector[:-1]

        # use only non-fixed parameters in simulate signal (we use the include property to do this)
        return s0 * self.dmipy_model.simulate_signal(dmipy_scheme, parameters[self.include[:-1]])

    def jacobian(self, scheme: DiffusionAcquisitionScheme) -> np.ndarray:

        # compute the baseline signal
        baseline = self.__call__(scheme)

        forward_diff = self._simulate_signals(self._parameter_vectors_forward, scheme)
        backward_diff = self._simulate_signals(self._parameter_vectors_backward, scheme)

        central_diff = forward_diff - backward_diff

        # reset parameters to original
        self.set_parameters_from_vector(self._parameter_baseline)

        # return jacobian
        jac = np.concatenate((central_diff * self._reciprocal_h, [baseline])).T
        return jac[:, self.include]

    def _simulate_signals(self, parameter_vectors: np.ndarray, scheme: DiffusionAcquisitionScheme) -> np.ndarray:
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

    def fit(self, scheme: DiffusionAcquisitionScheme, signal: np.ndarray, **fit_options) -> FittedDmipyModel:
        dmipy_scheme = convert_diffusion_scheme2dmipy_scheme(scheme)

        result = self._model.fit(dmipy_scheme, signal, **fit_options)

        return FittedDmipyModel(self, result)

    def set_initial_parameters(self, parameters: Dict[str, np.ndarray]) -> None:
        """
        Sets the initial guess on the _model attribute of this class.
        For this to work it is important that this method is called using parameter dict formatted using dmipy naming.
        :param parameters: Parameter dict formatted according to dmipy naming
        :return:
        """
        for name, value in parameters.items():
            if name in self._model.parameter_names:
                self._model.set_initial_guess_parameter(name, value)

    # TODO: refactor such that current parameter values are used
    @property
    def _dmipy_parameters(self) -> dict:
        """
        Gets a parameter dictionary that is compatible with the dmipy functions.
        :return:
        """
        parameters = {}
        # Extracting the parameters from the tissue model
        for model, model_name in zip(self.dmipy_model.models, self.dmipy_model.model_names):
            for parameter_name in model.parameter_names:
                value = np.array(getattr(model, parameter_name), dtype=np.float64, copy=True)
                parameters[model_name + parameter_name] = value

        # adding the partial volumes as well
        for pv_name in self._model.partial_volume_names:
            parameters[pv_name] = self[pv_name].value
        return parameters

    def dmipy_parameters2microtool_parameters(self, dmipy_pars: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Converts dictionary of dmipy parameter to microtool parameters

        :param dmipy_pars: dmipy parameter dictionary
        :return: microtool compatible parameter dictionary
        """
        dmipy_model = self._model

        mt_dict = {}
        for dmipy_name, dmipy_value in dmipy_pars.items():
            # decide how to add to dict based on the cardinality
            cardinality = dmipy_model.parameter_cardinality[dmipy_name]
            if cardinality == 1:
                # if we have scalar parameters microtool and dmipy treatment is the same
                mt_dict[dmipy_name] = dmipy_value
            else:
                # for vector parameters we split the to scalars and append an index to the name
                for i in range(cardinality):
                    mt_dict[dmipy_name + f'_{i}'] = dmipy_value[:, i]

        return mt_dict

    @property
    def dmipy_model(self):
        return self._model


class FittedDmipyModel(FittedModel):
    def __init__(self, dmipymodel: DmipyTissueModel, dmipyfitresult: FittedMultiCompartmentModel):
        # storing microtool model
        self._model = dmipymodel
        # storing dmipy fit result
        self.dmipyfitresult = dmipyfitresult

    @property
    def fitted_parameters(self) -> Dict[str, np.ndarray]:
        # Extracting the fitted parameter vector
        dmipy_pars = self.dmipyfitresult.fitted_parameters

        # Get the correct dict formatting from the tissuemodel
        return self._model.dmipy_parameters2microtool_parameters(dmipy_pars)

    @property
    def fit_information(self) -> Optional[dict]:
        return None


class AnalyticBall(DmipyTissueModel):
    """
    Quick and dirty inheritance of dmipytissue model. Purpose is for testing finite differences
    """

    def __init__(self, lambda_iso: float):
        model = G1Ball(lambda_iso)
        super().__init__(MultiCompartmentModel([model]))

    def jacobian_analytic(self, scheme: DiffusionAcquisitionScheme) -> np.ndarray:
        bvals = copy(scheme.b_values)

        # convert to SI units
        bvals *= 1e6

        S0 = self['S0'].value
        Diso = self['G1Ball_1_lambda_iso'].value

        # d S / d D_iso , d S / d S_0

        jac = np.array([- bvals * S0 * np.exp(-bvals * Diso), np.exp(-bvals * Diso)]).T
        return jac[:, self.include]


class CascadeDecorator(TissueModelDecoratorBase):
    """
    Use this decorator to add dmipytissuemodels that should be fitted before to initialize parameter values in more
    complex models
    """

    # just a reminder of the attribute names we are adding
    # (this is where we store the complex model)
    _original: DmipyTissueModel
    # storing the first model that we need to fit
    _simple_model: DmipyTissueModel
    # a map of the complex model parameter names as values and simple parameter names as keys
    _parameter_map: Dict[str, str]

    def __init__(self, complex_model: DmipyTissueModel, simple_model: DmipyTissueModel,
                 parameter_map: Dict[str, str]):
        # TODO check that simple model is indeed simpler than complex model

        super().__init__(complex_model)
        self._simple_model = deepcopy(simple_model)

        # TODO check parameter map for correct names etc
        self._parameter_map = parameter_map

    def fit(self, scheme: DiffusionAcquisitionScheme, signal: np.ndarray, **fit_options):
        # fit simple model
        simple_fit = self._simple_model.fit(scheme, signal, **fit_options)

        # extract the dmipyfitresult fitted parameters (i.e. the fit result in dmipy formatting)
        simple_parameters = simple_fit.dmipyfitresult.fitted_parameters

        # map parameters to initial values for complex model
        initial_values = self._name_map2value_map(simple_parameters, self._parameter_map)
        self._original.set_initial_parameters(initial_values)

        # fit complex model
        return self._original.fit(scheme, signal, **fit_options)

    @staticmethod
    def _name_map2value_map(fit_values: Dict[str, np.ndarray], name_map: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        :param fit_values: The fitted simple parameters
        :param name_map: The simple parameters that serve as
        :return:
        """
        value_map = {}
        for simple_name in name_map.keys():
            complex_name = name_map[simple_name]
            value_map[complex_name] = fit_values[simple_name]
        return value_map


class RelaxationDecorator(TissueModelDecoratorBase):
    # this is the variable we used for the original model in the decorator base (reminder)
    _original: DmipyTissueModel

    def __init__(self, model: DmipyTissueModel, T2: Union[List, np.ndarray, float]):
        super().__init__(model)

        # converting T2 to array
        if not isinstance(T2, np.ndarray):
            T2 = np.array(T2, dtype=float)

        # check the number of relaxivities with the number of models
        if T2.size != self._original.dmipy_model.N_models:
            raise ValueError("Specifiy relaxation for all compartments")

        # store as tissue_parameters
        if T2.size > 1:
            for i, value in enumerate(T2):
                self._original.update({"T2_relaxation_" + str(i): TissueParameter(value, 1.0)})
        else:
            self._original.update({"T2_relaxation": TissueParameter(float(T2), 1.0)})

        self._T2 = T2

    def __call__(self, scheme: DiffusionAcquisitionScheme):
        # making dmipy compatible scheme
        dmipy_scheme = convert_diffusion_scheme2dmipy_scheme(scheme)

        # Getting the original dmipy model
        dmipy_model = self._original.dmipy_model

        # Computing the signal that the individual comparments would generate given the acquisitionscheme
        S_compartment = compute_compartment_signals(dmipy_model, dmipy_scheme)

        # compute the decay caused by T2 relaxation for every compartment, shape (N_measure, N_comp)
        t2decay_factors = np.exp(- scheme.echo_times[:, np.newaxis] / self._T2)

        if dmipy_model.N_models == 1:
            # a single compartment does not have partial volumes so we just multiply by 1
            partial_volumes = 1.0
        else:
            # getting partial volumes as array
            partial_volumes = np.array([self._original[pv_name] for pv_name in dmipy_model.partial_volume_names])

        # multiply the computed signals of individual compartments by the T2-decay AND partial volumes!
        S_total = np.sum(partial_volumes * t2decay_factors * S_compartment, axis=-1)
        return S_total

    def fit(self, scheme: DiffusionAcquisitionScheme, signal: np.ndarray, **fit_options) -> FittedModel:
        # fit the original model
        naive_fit = super().fit(scheme, signal, **fit_options)

        # fit relaxation equation to the partial volume result

        raise NotImplementedError()

    def jacobian(self, scheme: DiffusionAcquisitionScheme) -> np.ndarray:
        # get the echo times from the scheme

        # co

        # convert the now extended parameter vector to the original partial volume only vector

        # update
        raise NotImplementedError()


def compute_compartment_signals(dmipy_model: MultiCompartmentModel,
                                dmipy_scheme: DmipyAcquisitionScheme) -> np.ndarray:
    """

    :param dmipy_model:
    :param dmipy_scheme:
    :return: signal array of shape (N_measure, N_model)
    """
    # array for storing signal from individual compartments
    S_compartment = np.zeros((dmipy_scheme.number_of_measurements, dmipy_model.N_models), dtype=float)

    # iterate over the individual models in the multi-compartment model
    for i, (model, model_name) in enumerate(zip(dmipy_model.models, dmipy_model.model_names)):

        # making a single compartment multi compartment (I know its stupid)
        # This is so we can generate signal from this
        single_compartment = MultiCompartmentModel([model])
        single_model_name = single_compartment.model_names[0]

        # Extracting the parameter dictionary
        parameters = {}
        for parameter_name in model.parameter_names:
            value = np.array(getattr(model, parameter_name), dtype=np.float64, copy=True)
            parameters[single_model_name + parameter_name] = value

        S_compartment[:, i] = single_compartment.simulate_signal(dmipy_scheme, parameters)

    return S_compartment

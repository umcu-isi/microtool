from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Dict, List, Optional, Union, Tuple, Sequence

import numpy as np
from dmipy.core.acquisition_scheme import DmipyAcquisitionScheme, acquisition_scheme_from_bvalues
from dmipy.core.fitted_modeling_framework import FittedMultiCompartmentModel
from dmipy.core.modeling_framework import ModelProperties as SingleDmipyModel
from dmipy.core.modeling_framework import MultiCompartmentModel

from .acquisition_scheme import DiffusionAcquisitionScheme, \
    DiffusionAcquisitionScheme_bval_dependency, DiffusionAcquisitionScheme_delta_dependency
from .constants import BASE_SIGNAL_KEY
from .scanner_parameters import ScannerParameters, default_scanner
from .tissue_model import TissueModel, TissueParameter, TissueModelDecorator, FittedModel
from .utils.unit_registry import unit

# dmipy wants b0 measurements but we are happy to handle schemes without b0 measuerements
warnings.filterwarnings('ignore', 'No b0 measurements were detected.*')


dmipy_to_microtool_name = {
    "bvalues": "B-Values",
    "delta": "DiffusionPulseWidth",
    "Delta": "DiffusionPulseInterval",
    "gradient_directions": "b-vectors",
    "gradient_strengths": "DiffusionPulseMagnitude",
}


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
            scales = model.parameter_scales[parameter_name]
            cardinality = model.parameter_cardinality[parameter_name]
            bounds = model.parameter_ranges[parameter_name]

            scaled_bounds = get_scaled_bounds(bounds, scales)
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
                    value=float(value[i]) if cardinality > 1 else float(value),
                    scale=scales[i] if cardinality > 1 else scales,
                    # non fixed or linked parameters can be included in the optimization.
                    optimize=not linked_or_fixed,
                    fit_bounds=(*scaled_bounds[i],)
                )

    return parameters


def get_scaled_bounds(bounds: Union[Tuple[List[float]], List[float]],
                      scales: Union[np.ndarray, float]) -> List[List[float]]:
    """
    Scales the bounds back to the order of magnitude scales.

    :param bounds:
    :param scales:
    :return: The bounds at the length scale of the parameter
    """
    if isinstance(scales, np.ndarray):
        # Arrays with bounds and scales.
        return [[bound[0] * scale, bound[1] * scale] for bound, scale in zip(bounds, scales)]
    else:
        return [[bounds[0] * scales, bounds[1] * scales]]


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
    if not isinstance(scheme, (DiffusionAcquisitionScheme, DiffusionAcquisitionScheme_bval_dependency, 
                               DiffusionAcquisitionScheme_delta_dependency)):
        raise TypeError(f"scheme is of type {type(scheme)}, we expected an {DiffusionAcquisitionScheme}")
    # note that dmipy has a different notion of echo times so they are not included in the conversion
    # Downcast pint-wrapped arrays to plain numpy arrays (during testing).
    return acquisition_scheme_from_bvalues(
        np.array(scheme.b_values, copy=False) * 1e6,  # Convert from s/mm² to s/m².
        np.array(scheme.b_vectors, copy=False),
        np.array(scheme.pulse_widths, copy=False),
        np.array(scheme.pulse_intervals, copy=False),
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
    b_values = scheme.bvalues * 1e-6 * unit('s/mm²')
    b_vectors = scheme.gradient_directions

    pulse_widths = scheme.delta * unit('s')
    pulse_intervals = scheme.Delta * unit('s')
    if scheme.TE is None:
        return DiffusionAcquisitionScheme.from_bvals(b_values, b_vectors, pulse_widths, pulse_intervals,
                                                     scanner_parameters=scanner_parameters)
    else:
        echo_times = scheme.TE * unit('s')
        return DiffusionAcquisitionScheme.from_bvals(b_values, b_vectors, pulse_widths, pulse_intervals,
                                                     echo_times, scanner_parameters=scanner_parameters)


class DmipyMultiTissueModel(TissueModel):
    """
    Wrapper for the MultiCompartment models used by dmipy. Note that the parameters need to be initialized in the
    dmipy model otherwise a value error is raised.
    """
    _model: MultiCompartmentModel  # Reminder that we store the dmipy multicompartment model in this attribute

    def __init__(self,
                 model: Union[MultiCompartmentModel, SingleDmipyModel, Sequence[SingleDmipyModel]],
                 volume_fractions: Union[Sequence[float], float] = None):
        """
        :param model: Either a MultiCompartment model, a single Dmipy model or a sequence of Dmipy models.
            Models created from Dmipy toolbox are to be stored as MultiCompartment instances as to utilize
            the associated functionalities for signal generation, fitting and more, which are absent in Model classes.
        :param volume_fractions: The relative volume fractions of the models (order in the same way you initialized the
                                 multi-compartment model)
        """
        if not isinstance(model, MultiCompartmentModel):
            if isinstance(model, list):
                model = MultiCompartmentModel(model)
            else:
                model = MultiCompartmentModel([model])

        # Extract the tissue parameters from individual models and convert to 'scalars'. (makes parameter dict)
        parameters = get_parameters(model)

        # -------------------- Set up volume fractions if there are multiple models
        if model.N_models > 1:
            if volume_fractions is None:
                raise ValueError("Provide volume fractions if composite tissue models are used.")
            # Get the ordered partial volume names from the model
            vf_keys = model.partial_volume_names
            # check if the volume_fractions match the length of this dictionary
            if len(volume_fractions) != len(vf_keys):
                raise ValueError("Number of volume fractions does not match the number of models.")
            if np.array(volume_fractions).sum() != 1:
                raise ValueError("Volume fractions do not sum up to 1.")
            # Including the volume fractions as TissueParameters to the DmipyMultiTissueModel
            for i, key in enumerate(vf_keys):
                parameters.update({key: TissueParameter(value=volume_fractions[i], scale=1., fit_bounds=(0.0, 1.0))})

        # Add S0 as a tissue parameter (to be excluded in parameters extraction etc.)
        parameters.update({BASE_SIGNAL_KEY: TissueParameter(value=1.0, scale=1.0, optimize=False, fit_flag=False)})
        self._model = model
        super().__init__(parameters)

    def __call__(self, scheme: DiffusionAcquisitionScheme) -> np.ndarray:
        # Convert the microtool scheme to a dmipy compatible scheme.
        dmipy_scheme = convert_diffusion_scheme2dmipy_scheme(scheme)

        # Computing the signal that the individual compartments would generate given the acquisition scheme
        signals = compute_compartment_signals(self._model, dmipy_scheme)

        if self._model.N_models == 1:
            # a single compartment does not have partial volumes so we just multiply by 1
            partial_volumes = 1.0
        else:
            # getting partial volumes as array
            partial_volumes = np.array([self[pv_name].value for pv_name in self._model.partial_volume_names])

        # multiply the computed signals of individual compartments by the T2-decay AND partial volumes!
        return self[BASE_SIGNAL_KEY].value * np.sum(partial_volumes * signals, axis=-1)

    @property
    def dmipy_model(self) -> MultiCompartmentModel:
        return self._model

    def set_parameters_from_vector(self, new_parameter_values: np.ndarray) -> None:
        # doing the microtool update
        super().set_parameters_from_vector(new_parameter_values)
        self._dmipy_set_parameters(new_parameter_values)

    def set_fit_parameters(self, new_values: Union[np.ndarray, dict]) -> None:
        super().set_fit_parameters(new_values)
        # Also we need to set the parameters on the dmipy model for the signal simulation to work
        full_vector = self.parameter_vector
        full_vector[self.include_fit] = new_values
        # idee: maak een volledige vector waarbij de oude waarden gekopieerd worden voor dit object.
        self._dmipy_set_parameters(full_vector[:-self._model.N_models])

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

    def _dmipy_set_parameters(self, vector: np.ndarray) -> None:
        """
        Sets the correct value for the dmipy parameters on the underlying dmipy model

        :param vector: the vector of the dmipy parameters only!
        :return: nothing
        """
        k = 0
        for model, model_name in zip(self._model.models, self._model.model_names):
            for parameter_name in model.parameter_names:
                par_size = model.parameter_cardinality[parameter_name]
                setattr(model, parameter_name, vector[k:(k + par_size)])
                k += par_size

    def _dmipy_fix_parameters(self, fix_parameter: str, fix_value: float) -> None:
        """
        Wrapper for Dmipy's own MultiComparmentModel function that sets fixed model parameters

        :fix parameter: string of parameter to fix
        :fix_value: value to fix the paramter to
        :return: nothing
        """
        self._model.set_fixed_parameter(fix_parameter, fix_value)
        
        parameters = get_parameters(self._model)
        super().__init__(parameters)

    def get_dependencies(self) -> list:
        """
        Obtains model dependencies from Dmipy package as defined by each model class
        
        :return: list with model parameter dependencies     
        """
        requirements = []     
        for model in self._model.models:
            # Obtain from dmipy model the required acquisition parameters
            parameters = model._required_acquisition_parameters           
            translated_params = [dmipy_to_microtool_name[param] for param in parameters]
    
            requirements = requirements + translated_params
            
        requirement_list = list(set(requirements))  # Remove duplicates and translate back to list
        
        return requirement_list

    def check_dependencies(self, scheme: DiffusionAcquisitionScheme):
        """
        Method for consistency check-up between model requirements and defined scheme parameters
    
        """
        model_requirements = self.get_dependencies()
            
        # If any of these parameters is not set for optimization, raise warning
        for param in model_requirements:

            # B-values and b-vectors computed from established parameter relations.
            if param in ['B-Values', 'b-vectors']:
                continue
            elif scheme[param].fixed:
                warnings.warn(f"Parameter {param} is fixed, but it should be optimized for the model.")
                    
        return model_requirements


class FittedDmipyModel(FittedModel):
    def __init__(self, dmipymodel: DmipyMultiTissueModel, dmipyfitresult: FittedMultiCompartmentModel):
        # storing microtool model
        self._model = dmipymodel
        # storing dmipy fit result
        self.dmipyfitresult = dmipyfitresult

    @property
    def fitted_parameters(self) -> Dict[str, np.ndarray]:
        # TODO: self.dmipyfitresult is a FittedMultiCompartmentModel, so why are there dmipy parameter names in there?
        # Extracting the fitted parameter vector
        dmipy_pars = self.dmipyfitresult.fitted_parameters

        # Get the correct dict formatting from the tissue model.
        return get_microtool_parameters(self._model.dmipy_model, dmipy_pars)

    @property
    def print_fit_information(self) -> Optional[dict]:
        return None


# TODO: Do we need this?
class CascadeFitDmipy(TissueModelDecorator):
    """
    Use this decorator to add dmipytissuemodels that should be fitted before to initialize parameter values in more
    complex models
    """

    # just a reminder of the attribute names we are adding
    # (this is where we store the complex model)
    _original: DmipyMultiTissueModel
    # storing the first model that we need to fit
    _simple_model: DmipyMultiTissueModel
    # a map of the complex model parameter names as values and simple parameter names as keys
    _parameter_map: Dict[str, str]

    def __init__(self, complex_model: DmipyMultiTissueModel, simple_model: DmipyMultiTissueModel,
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


def compute_compartment_signals(dmipy_model: MultiCompartmentModel,
                                dmipy_scheme: DmipyAcquisitionScheme) -> np.ndarray:
    """

    :param dmipy_model: a dmipy multi-compartment model
    :param dmipy_scheme: a dmipy acquisition scheme
    :return: signal array of shape (N_measures, N_models)
    """
    # array for storing signal from individual compartments
    signals = np.zeros((dmipy_scheme.number_of_measurements, dmipy_model.N_models), dtype=float)

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

        signals[:, i] = single_compartment.simulate_signal(dmipy_scheme, parameters)

    return signals


def get_microtool_parameters(model: MultiCompartmentModel, dmipy_pars: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Converts dictionary of dmipy parameters to microtool parameters

    :param model: a dmipy multi-compartment model
    :param dmipy_pars: dmipy parameter dictionary
    :return: microtool compatible parameter dictionary
    """
    parameters = {}
    for dmipy_name, dmipy_value in dmipy_pars.items():
        # decide how to add to dict based on the cardinality
        cardinality = model.parameter_cardinality[dmipy_name]
        if cardinality == 1:
            # if we have scalar parameters microtool and dmipy treatment is the same
            parameters[dmipy_name] = dmipy_value
        else:
            # for vector parameters we split to scalars and append an index to the name
            for i in range(cardinality):
                parameters[dmipy_name + f'_{i}'] = dmipy_value[:, i]

    return parameters

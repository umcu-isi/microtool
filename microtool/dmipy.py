from __future__ import annotations

import warnings
from copy import copy, deepcopy
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
from dmipy.core.acquisition_scheme import DmipyAcquisitionScheme, acquisition_scheme_from_bvalues
from dmipy.core.fitted_modeling_framework import FittedMultiCompartmentModel
from dmipy.core.modeling_framework import ModelProperties as SingleDmipyModel
from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.signal_models.gaussian_models import G1Ball

from microtool.acquisition_scheme import DiffusionAcquisitionScheme
from microtool.constants import BASE_SIGNAL_KEY
from microtool.scanner_parameters import ScannerParameters, default_scanner
from microtool.tissue_model import TissueModel, TissueParameter, TissueModelDecorator, FittedModel

# dmipy wants b0 measurements but we are happy to handle schemes without b0 measuerements
warnings.filterwarnings('ignore', 'No b0 measurements were detected.*')


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
                    value=value[i] if cardinality > 1 else value,
                    scale=scales[i] if cardinality > 1 else scales,
                    # non fixed or linked parameters can be included in the optimization.
                    optimize=not linked_or_fixed,
                    fit_bounds=(*scaled_bounds[i],)
                )

    return parameters


def get_scaled_bounds(dmipy_bounds: Tuple[List[float]], scales: Union[List[float], float]):
    """
    Scales the bounds back to the order of magnitude scales.

    :param dmipy_bounds:
    :param scales:
    :return: The bounds at the length scale of the parameter
    """
    bounds_lst = list(dmipy_bounds)
    scaled_bounds = []
    if isinstance(scales, np.ndarray):
        for bound, scale in zip(bounds_lst, scales):
            scaled_bounds.append([bound[0] * scale, bound[1] * scale])
    else:
        scale = scales
        scaled_bounds.append([bounds_lst[0] * scale, bounds_lst[1] * scale])
    return scaled_bounds


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
        scheme.pulse_widths,
        scheme.pulse_intervals,
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

    pulse_widths = scheme.delta
    pulse_intervals = scheme.Delta
    echo_times_SI = scheme.TE
    if echo_times_SI is None:
        return DiffusionAcquisitionScheme.from_bvals(b_values, b_vectors, pulse_widths, pulse_intervals,
                                                     scan_parameters=scanner_parameters)
    else:
        return DiffusionAcquisitionScheme.from_bvals(b_values, b_vectors, pulse_widths, pulse_intervals,
                                                     echo_times_SI,
                                                     scan_parameters=scanner_parameters)


class DmipyTissueModel(TissueModel):
    """
    Wrapper for the MultiCompartment models used by dmipy. Note that the parameters need to be initialized in the
    dmipy model otherwise a value error is raised.
    """
    _model: MultiCompartmentModel  # Reminder that we store the dmipy multicompartment model in this attribute

    def __init__(self, dmipy_models: Union[Sequence[SingleDmipyModel], SingleDmipyModel], volume_fractions: Union[Sequence[float], float] = None):
        """

        :param model: MultiCompartment model
            Models created from Dmipy toolbox are to be stored as MultiCompartment instances as to utilize
            the associated functionalities for signal generation, fitting and more, which are absent in Model classes.
        :param volume_fractions: The relative volume fractions of the models (order in the same way you initialized the
                                 multicompartment model)
        """

        if not isinstance(dmipy_models, list):
            dmipy_models = [dmipy_models]

        model = MultiCompartmentModel(dmipy_models)

        # Extract the tissue parameters from individual models and convert to 'scalars'. (makes parameter dict)
        parameters = get_parameters(model)

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
                parameters.update({key: TissueParameter(value=volume_fractions[i], scale=1., fit_bounds=(0.0, 1.0))})

        # Add S0 as a tissue parameter (to be excluded in parameters extraction etc.)
        parameters.update({BASE_SIGNAL_KEY: TissueParameter(value=1.0, scale=1.0, optimize=False, fit_flag=False)})
        self._model = model
        super().__init__(parameters)

    def __call__(self, scheme: DiffusionAcquisitionScheme) -> np.ndarray:
        # converting the microtool scheme to a dmipy compatible scheme
        dmipy_scheme = convert_diffusion_scheme2dmipy_scheme(scheme)

        # Getting the original dmipy model
        dmipy_model = self.dmipy_model

        # Computing the signal that the individual comparments would generate given the acquisitionscheme
        S_compartment = compute_compartment_signals(dmipy_model, dmipy_scheme)

        if dmipy_model.N_models == 1:
            # a single compartment does not have partial volumes so we just multiply by 1
            partial_volumes = 1.0
        else:
            # getting partial volumes as array
            partial_volumes = np.array([self[pv_name].value for pv_name in dmipy_model.partial_volume_names])

        # multiply the computed signals of individual compartments by the T2-decay AND partial volumes!
        S_total = self[BASE_SIGNAL_KEY].value * np.sum(partial_volumes * S_compartment, axis=-1)
        return S_total

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
        Sets as fixed desired dmipy model parameters 

        :fix parameter: string of parameter to fix
        :fix_value: value to fix the paramter to
        :return: nothing
        """
        dmipy_models = self.dmipy_model
        dmipy_models.set_fixed_parameter(fix_parameter, fix_value)
        
        parameters = get_parameters(dmipy_models)
        super().__init__(parameters)
        
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

    def check_dependecies(self, scheme: DiffusionAcquisitionScheme):
        """
        Method for consistency check-up between model requirements and defined scheme parameters
    
        """          
        dmipy_model = self.dmipy_model
        
        for model in enumerate(dmipy_model.models):
            #Obtain from dmipy model the required acquisition parameters
            required = model._required_acquisition_parameters
            
            #If any of these parameters is not set for optimization, raise warning
            for param in required:
                #Translate dmipy acquisition parameter name to microtool nomenclature
                param_name = dmipy2micotrool_dictionary_translation(param)

                #B-values and b-vectors computed from established parameter relations.
                if param_name in ['B-Values', 'b-vectors']:
                    continue
                elif scheme._are_fixed([param_name]):
                    warnings.warn(f"Parameter {param} should be optimized for {model} model.")

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
    def print_fit_information(self) -> Optional[dict]:
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
        TE = scheme.echo_times
        # convert to SI units
        bvals *= 1e6

        S0 = self[BASE_SIGNAL_KEY].value
        Diso = self['G1Ball_1_lambda_iso'].value

        # the signal S = S_0 * e^{-T_E / T_2} * e^{-b * D}
        S = S0 * np.exp(-bvals * Diso)
        s_diso = - bvals * S

        # d S / d D_iso , d S / d S_0
        jac = np.array([s_diso, S]).T
        return jac[:, self.include_optimize]


class CascadeFitDmipy(TissueModelDecorator):
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

def dmipy2micotrool_dictionary_translation(parameter: str) -> str:
    
    to_microtool_name = {
    "bvalues": "B-Values",
    "delta": "DiffusionPulseWidth",
    "Delta": "DiffusionPulseInterval",
    "gradient_directions": "b-vectors",
    "gradient_strengths": "DiffusionPulseMagnitude",
    }

    microtool_param = to_microtool_name[parameter]
    
    return microtool_param
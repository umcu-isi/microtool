from __future__ import annotations

from copy import copy
from typing import Dict, List, Optional

import numpy as np
from dmipy.core.acquisition_scheme import DmipyAcquisitionScheme, acquisition_scheme_from_bvalues
from dmipy.core.fitted_modeling_framework import FittedMultiCompartmentModel
from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.signal_models.gaussian_models import G1Ball

from microtool.acquisition_scheme import DiffusionAcquisitionScheme
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

    :param scheme: DiffusionAcquisitionScheme
    :return: DmipyAcquisitionScheme
    """
    if not isinstance(scheme, DiffusionAcquisitionScheme):
        raise TypeError(f"scheme is of type {type(scheme)}, we expected an {DiffusionAcquisitionScheme}")

    return acquisition_scheme_from_bvalues(
        scheme.b_values * 1e6,  # Convert from s/mm² to s/m².
        scheme.b_vectors,
        scheme.pulse_widths * 1e-3,  # Convert from ms to s.
        scheme.pulse_intervals * 1e-3,  # Convert from ms to s.
    )


def convert_dmipy_scheme2diffusion_scheme(scheme: DmipyAcquisitionScheme) -> DiffusionAcquisitionScheme:
    """
    Takes a scheme from the dmipy toolbox and converts to scheme from microtool toolbox. We convert units since
    microtool has non SI units.

    :param scheme: DmipyAcquisitionScheme
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
    return DiffusionAcquisitionScheme(b_values, b_vectors, pulse_widths, pulse_intervals)


class DmipyTissueModel(TissueModel):
    """
    Wrapper for the MultiCompartment models used by dmipy. Note that the parameters need to be initialized in the
    dmipy model otherwise a value error is raised.
    """

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
        return s0 * self._model.simulate_signal(dmipy_scheme, self._dmipy_parameters)

    def jacobian(self, scheme: DiffusionAcquisitionScheme) -> np.ndarray:
        dmipy_scheme = convert_diffusion_scheme2dmipy_scheme(scheme)

        # Evaluate the dmipy model on the baseline and on the parameter vectors with finite differences.
        s0 = self['S0'].value
        # baseline signal for UNvaried tissueparameters
        baseline = self._model.simulate_signal(dmipy_scheme, self._parameter_baseline)
        # d S for all the different tissue parameters
        forward_diff = self._model.simulate_signal(dmipy_scheme, self._parameter_vectors_forward)
        backward_diff = self._model.simulate_signal(dmipy_scheme, self._parameter_vectors_backward)
        central_diff = s0 * (forward_diff - backward_diff)

        # Divide by the finite differences to obtain the derivatives (central difference method),
        # and concatenate the derivatives for S0 (i.e. the baseline signal).
        jac = np.concatenate((central_diff * self._reciprocal_h, [baseline])).T
        return jac

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

    @property
    def _dmipy_parameters(self) -> dict:
        """
        Gets a parameter dictionary that is compatible with the dmipy functions.
        :return:
        """
        parameters = {}
        # Extracting the parameters from the tissue model
        for model, model_name in zip(self._model.models, self._model.model_names):
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
        return jac


class CascadeDecorator(TissueModelDecoratorBase):
    """
    Use this decorator to add dmipytissuemodels that should be fitted before to initialize parameter values in more
    complex models
    """

    # just a reminder of the attribute names we are adding
    # (this is where we store the complex model)
    _original: DmipyTissueModel = None
    # storing the first model that we need to fit
    _simple_model: DmipyTissueModel = None
    # a map of the complex model parameter names as values and simple parameter names as keys
    _parameter_map: Dict[str, str] = None

    def __init__(self, complex_model: DmipyTissueModel, simple_model: DmipyTissueModel, parameter_map: Dict[str, str]):
        # TODO check that simple model is indeed simpler than complex model

        super().__init__(complex_model)
        self._simple_model = simple_model

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

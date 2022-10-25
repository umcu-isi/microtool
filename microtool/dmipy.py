from copy import copy
from typing import Dict, Union

import numpy as np
from dmipy.core.acquisition_scheme import DmipyAcquisitionScheme, acquisition_scheme_from_bvalues
from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.signal_models.gaussian_models import G1Ball

from microtool.acquisition_scheme import DiffusionAcquisitionScheme
from microtool.tissue_model import TissueModel, TissueParameter


def get_parameters(diffusion_model: MultiCompartmentModel) -> Dict[str, TissueParameter]:
    """
    Compiles a list of all tissue parameters present in the given multi-compartment model.

    :param diffusion_model: A dmipy multi-compartment model.
    :return: A list with tissue parameters.
    """
    # Iterate over all tissue models in the MultiCompartmentModel.
    parameters = {}
    for model, model_name in zip(diffusion_model.models, diffusion_model.model_names):
        # Iterate over all scalar and vector parameters in the tissue model.
        for parameter_name in model.parameter_names:
            value = np.array(getattr(model, parameter_name), dtype=np.float64, copy=True)
            scale = model.parameter_scales[parameter_name]
            cardinality = model.parameter_cardinality[parameter_name]

            if value is None:
                raise ValueError(f'Parameter {parameter_name} of model {model_name} has no value.')
            if np.any(np.isnan(value)):
                raise ValueError(f'Parameter {parameter_name} of model {model_name} has a nan value.')
            # Iterate over vector parameters and add their elements as scalar tissue parameters.
            for i in range(cardinality):
                index_postfix = '' if cardinality == 1 else f'_{i}'
                parameters[model_name + parameter_name + index_postfix] = TissueParameter(
                    value=value[i] if cardinality > 1 else value,
                    scale=scale[i] if cardinality > 1 else scale,
                )

    return parameters


def convert_acquisition_scheme(
        scheme: Union[DiffusionAcquisitionScheme, DmipyAcquisitionScheme]) -> DmipyAcquisitionScheme:
    # Create a dmipy acquisition scheme.
    if isinstance(scheme, DmipyAcquisitionScheme):
        return scheme
    else:
        return acquisition_scheme_from_bvalues(
            scheme.b_values * 1e6,  # Convert from s/mm² to s/m².
            scheme.b_vectors,
            scheme.pulse_widths * 1e-3,  # Convert from ms to s.
            scheme.pulse_intervals * 1e-3,  # Convert from ms to s.
        )


class DmipyAcquisitionSchemeWrapper(DiffusionAcquisitionScheme):
    """
    Class wrapper for pure dmipy acquisition schemes
    """

    def __init__(self, scheme: DmipyAcquisitionScheme):
        self._scheme = scheme
        # convert to s/mm^2 from s/m^2
        b_values = scheme.bvalues * 1e-6
        b_vectors = scheme.gradient_directions
        # convert to ms from s
        pulse_widths = scheme.delta * 1e3
        pulse_intervals = scheme.Delta * 1e3
        super().__init__(b_values, b_vectors, pulse_widths, pulse_intervals)


class DmipyTissueModel(TissueModel):
    """
    Wrapper for the MultiCompartment models used by dmipy. Note that the parameters need to be initialized in the
    dmipy model otherwise a value error is raised.
    """

    def __init__(self, model: MultiCompartmentModel, volume_fractions: np.ndarray = None):
        """

        :param model: MultiCompartment model
        :param volume_fractions: The relative volume fractions of the models (order in the same way you initialized the
                                 multicompartment model)
        """
        super().__init__()
        # Extract the scalar tissue parameters from individual models.
        self.update(get_parameters(model))

        # Set up volume fractions if there are multiple models
        if model.N_models > 1:
            if volume_fractions is None:
                raise ValueError("Provide volume fractions if composite tissuemodels are used.")
            # Get the ordered partial volume names from the model
            vf_keys = model.partial_volume_names
            # check if the volume_fractions match the length of this dictionary
            if len(volume_fractions) != len(vf_keys):
                raise ValueError("Not enough volume fractions provided for the number of models.")
            if np.sum(volume_fractions) != 1:
                raise ValueError("Provide volume fractions do not sum to 1.")
            # Including the volume fractions as TissueParameters to the DmipyTissueModel
            for i, key in enumerate(vf_keys):
                self.update({key: TissueParameter(value=volume_fractions[i], scale=1., optimize=False)})

        # Add S0 as a tissue parameter (to be excluded in parameters extraction etc.)
        self.update({'S0': TissueParameter(value=1.0, scale=1.0, optimize=False)})
        self._model = model

        # JACOBIAN HELPER VARIABLES ----------------------------

        # Get the baseline parameter vector, but don't include S0.
        self._parameter_baseline = np.array([parameter.value for parameter in self.values()])[:-1]

        # Calculate finite differences and corresponding parameter vectors for calculating derivatives.
        step_size = 1e-6
        h = np.array([parameter.scale * step_size for parameter in self.values()])[:-1]
        self._parameter_vectors = self._parameter_baseline + np.diag(h)
        self._reciprocal_h = (1 / h).reshape(-1, 1)

    def __call__(self, scheme: DiffusionAcquisitionScheme) -> np.ndarray:
        dmipy_scheme = convert_acquisition_scheme(scheme)

        # Evaluate the dmipy model.
        s0 = self['S0'].value
        return s0 * self._model.simulate_signal(dmipy_scheme, self._dmipy_parameters)

    def jacobian(self, scheme: DiffusionAcquisitionScheme) -> np.ndarray:
        dmipy_scheme = convert_acquisition_scheme(scheme)

        # Evaluate the dmipy model on the baseline and on the parameter vectors with finite differences.
        s0 = self['S0'].value
        # baseline signal for UNvaried tissueparameters
        baseline = self._model.simulate_signal(dmipy_scheme, self._parameter_baseline)
        # d S for all the different tissue parameters
        differences = s0 * (self._model.simulate_signal(dmipy_scheme, self._parameter_vectors) - baseline)

        # Divide by the finite differences to obtain the derivatives,
        # and concatenate the derivatives for S0 (i.e. the baseline signal).
        jac = np.concatenate((differences * self._reciprocal_h, [baseline])).T
        return jac

    def fit(self, scheme: DmipyAcquisitionScheme, noisy_signal: np.ndarray, **fit_options):
        dmipy_scheme = convert_acquisition_scheme(scheme)

        result = self._model.fit(dmipy_scheme, noisy_signal, **fit_options)
        return result

    def set_initial_parameters(self, parameters: Dict[str, np.ndarray]) -> None:
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


class AnalyticBall(DmipyTissueModel):
    """
    Quick and dirty inheritance of dmipytissue model. Purpose is simply to overwrite the jacobian
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

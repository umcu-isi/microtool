from typing import Dict, Union

import numpy as np
from dmipy.core.acquisition_scheme import DmipyAcquisitionScheme, acquisition_scheme_from_bvalues
from dmipy.core.modeling_framework import MultiCompartmentModel

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

    # Add S0 as a tissue parameter.
    parameters['S0'] = TissueParameter(value=1.0, scale=1.0, optimize=False)

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


# TODO: implement fit method
class DmipyTissueModel(TissueModel):
    """
    Wrapper for the MultiCompartment models used by dmipy. Note that the parameters need to be initialized in the
    dmipy model otherwise a value error is raised.
    """

    def __init__(self, model: MultiCompartmentModel, volume_fractions: np.ndarray = None):
        """

        :param model: MultiCompartment model
        :param volume_fractions: The relative volume fractions of the models (order in the sameway you initialized the
        multicompartment model)
        """
        super().__init__()

        # Extract the scalar tissue parameters.
        self.update(get_parameters(model))

        if model.N_models > 1:
            if volume_fractions is None:
                raise ValueError("Provide volume fractions of composite tissuemodels are used.")
            # Get the ordered partial volume names from the model
            vf_keys = model.partial_volume_names
            # check if the volume_fractions match the length of this dictionary
            if len(volume_fractions) != len(vf_keys):
                raise ValueError("Not enough volume fractions provided for the number of models.")
            # Including the volume fractions as TissueParameters to the DmipyTissueModel
            for i, key in enumerate(vf_keys):
                self.update({key: TissueParameter(value=volume_fractions[i], scale=1., optimize=False)})

        self._model = model

        # Get the baseline parameter vector, but don't include S0.
        self._parameter_baseline = np.array([parameter.value for parameter in self.values()])[:-1]

        # Calculate finite differences and corresponding parameter vectors for calculating derivatives.
        h = np.array([parameter.scale * 1e-6 for parameter in self.values()])[:-1]
        self._parameter_vectors = self._parameter_baseline + np.diag(h)
        self._reciprocal_h = (1 / h).reshape(-1, 1)

    def __call__(self, scheme: DiffusionAcquisitionScheme) -> np.ndarray:
        dmipy_scheme = convert_acquisition_scheme(scheme)

        # Evaluate the dmipy model.
        s0 = self['S0'].value
        return s0 * self._model.simulate_signal(dmipy_scheme, self._parameter_baseline)

    def jacobian(self, scheme: DiffusionAcquisitionScheme) -> np.ndarray:
        dmipy_scheme = convert_acquisition_scheme(scheme)

        # Evaluate the dmipy model on the baseline and on the parameter vectors with finite differences.
        s0 = self['S0'].value
        baseline = self._model.simulate_signal(dmipy_scheme, self._parameter_baseline)
        differences = s0 * (self._model.simulate_signal(dmipy_scheme, self._parameter_vectors) - baseline)

        # Divide by the finite differences to obtain the derivatives, and add the derivatives for S0.
        return np.concatenate((differences * self._reciprocal_h, [baseline])).T

    def fit(self, scheme: DmipyAcquisitionScheme, noisy_signal: np.ndarray, **fit_options):
        dmipy_scheme = convert_acquisition_scheme(scheme)
        result = self._model.fit(dmipy_scheme, noisy_signal, fit_options)
        # TODO: use tissuemodel wrapper for output Note that some of the parameters are not included in the fitting
        #  and hence will not be returned when calling fitted_parameters on FittedMultiCompartmentModel

        return result

    # class FittedDmipyTissueModel:
    #     def __init__(self, model:DmipyTissueModel, fitted_parameters: Dict[str, np.ndarray]):
    #         self._model = model
    #         self.fitted_parameter_dmipy = fitted_parameters
    #
    #     @property
    #     def fitted_parameters(self) -> Dict[str, float]:
    #         pass

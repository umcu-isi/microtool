from typing import Dict

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

            # Iterate over vector parameters and add their elements as scalar tissue parameters.
            for i in range(cardinality):
                index_postfix = '' if cardinality == 1 else f'_{i}'
                parameters[model_name + parameter_name + index_postfix] = TissueParameter(
                    value=value[i] if cardinality > 1 else value,
                    scale=scale[i] if cardinality > 1 else scale,
                )

    # Add S0 as a tissue parameter.
    parameters['s0'] = TissueParameter(value=1.0, scale=1.0, use=False)

    return parameters


def convert_acquisition_scheme(scheme: DiffusionAcquisitionScheme) -> DmipyAcquisitionScheme:
    b_values = scheme.get_b_values()
    b_vectors = scheme.get_b_vectors()
    pulse_widths = scheme.get_pulse_widths()
    pulse_intervals = scheme.get_pulse_intervals()

    # Create a dmipy acquisition scheme. Convert b-values from s/mm² to s/m².
    return acquisition_scheme_from_bvalues(b_values * 1e6, b_vectors, pulse_widths, pulse_intervals)


class DmipyTissueModel(TissueModel):
    def __init__(self, model: MultiCompartmentModel):
        # Extract the scalar tissue parameters.
        self._model = model
        self._parameters = get_parameters(model)

        # Get the baseline parameter vector, but don't include S0.
        self._parameter_baseline = np.array([parameter.value for parameter in self._parameters.values()])[:-1]

        # Calculate finite differences and corresponding parameter vectors for calculating derivatives.
        h = np.array([parameter.scale * 1e-6 for parameter in self._parameters.values()])[:-1]
        self._parameter_vectors = self._parameter_baseline + np.diag(h)
        self._reciprocal_h = (1 / h).reshape(-1, 1)

    def __call__(self, scheme: DiffusionAcquisitionScheme) -> np.ndarray:
        dmipy_scheme = convert_acquisition_scheme(scheme)

        # Evaluate the dmipy model.
        s0 = self._parameters['s0'].value
        return s0 * self._model.simulate_signal(dmipy_scheme, self._parameter_baseline)

    def jacobian(self, scheme: DiffusionAcquisitionScheme) -> np.ndarray:
        dmipy_scheme = convert_acquisition_scheme(scheme)

        # Evaluate the dmipy model on the baseline and on the parameter vectors with finite differences.
        s0 = self._parameters['s0'].value
        baseline = self._model.simulate_signal(dmipy_scheme, self._parameter_baseline)
        differences = s0 * (self._model.simulate_signal(dmipy_scheme, self._parameter_vectors) - baseline)

        # Divide by the finite differences to obtain the derivatives, and add the derivatives for S0.
        return np.concatenate((differences * self._reciprocal_h, [baseline])).T

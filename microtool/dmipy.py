from dataclasses import dataclass
from typing import List

import numpy as np
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dmipy.core.modeling_framework import MultiCompartmentModel

from microtool.tissue_model import DiffusionModel


@dataclass
class TissueParameter:
    """
    Defines a dmipy scalar tissue parameter and its properties.
    """
    model: str
    name: str
    scalar: bool
    index: int
    value: float
    scale: float


def get_parameters(diffusion_model: MultiCompartmentModel) -> List[TissueParameter]:
    """
    Compiles a list of all tissue parameters present in the given multi-compartment model.

    :param diffusion_model: A dmipy multi-compartment model.
    :return: A list with tissue parameters.
    """
    # Iterate over all tissue models in the MultiCompartmentModel.
    parameters = []
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
                parameters.append(
                    TissueParameter(
                        model=model_name,
                        name=parameter_name,
                        scalar=cardinality == 1,
                        index=i,
                        value=value[i] if cardinality > 1 else value,
                        scale=scale[i] if cardinality > 1 else scale,
                    )
                )

    return parameters


class DmipyDiffusionModel(DiffusionModel):
    """
    Wraps a dmipy MultiCompartmentModel as a MICROtool DiffusionModel.

    :param diffusion_model: A dmipy MultiCompartmentModel
    """
    def __init__(self, diffusion_model: MultiCompartmentModel):
        super().__init__()

        # Extract the scalar tissue parameters.
        self._diffusion_model = diffusion_model
        self._parameters = get_parameters(diffusion_model)

        # Get the baseline parameter vector.
        self._parameter_baseline = np.array([parameter.value for parameter in self._parameters])

        # Calculate finite differences and corresponding parameter vectors for calculating derivatives.
        h = np.array([parameter.scale * 1e-6 for parameter in self._parameters])
        self._parameter_vectors = self._parameter_baseline + np.diag(h)
        self._reciprocal_h = (1 / h).reshape(-1, 1)

    def __call__(self,
                 b_values: np.ndarray,
                 b_vectors: np.ndarray,
                 pulse_widths: np.ndarray,
                 pulse_intervals: np.ndarray) -> np.ndarray:
        # Create a dmipy acquisition scheme. Convert b-values from s/mm² to s/m².
        acquisition_scheme = acquisition_scheme_from_bvalues(b_values * 1e6, b_vectors, pulse_widths, pulse_intervals)

        # Evaluate the dmipy model.
        return self._diffusion_model.simulate_signal(acquisition_scheme, self._parameter_baseline)

    def jacobian(self,
                 b_values: np.ndarray,
                 b_vectors: np.ndarray,
                 pulse_widths: np.ndarray,
                 pulse_intervals: np.ndarray):
        # Create a dmipy acquisition scheme. Convert b-values from s/mm² to s/m².
        acquisition_scheme = acquisition_scheme_from_bvalues(b_values * 1e6, b_vectors, pulse_widths, pulse_intervals)

        # Evaluate the dmipy model on the baseline and on the parameter vectors with finite differences.
        baseline = self._diffusion_model.simulate_signal(acquisition_scheme, self._parameter_baseline)
        differences = self._diffusion_model.simulate_signal(acquisition_scheme, self._parameter_vectors) - baseline

        # Divide by the finite differences to obtain the derivatives.
        jac = (differences * self._reciprocal_h).T

        return jac

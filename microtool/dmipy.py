from dataclasses import dataclass
from typing import List, Union

import numpy as np
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues, DmipyAcquisitionScheme
from dmipy.core.modeling_framework import MultiCompartmentModel

from .acquisition_scheme import AcquisitionScheme


@dataclass
class TissueParameter:
    model: str
    name: str
    scalar: bool
    index: int
    value: float
    scale: float


def get_parameters(tissue_model: MultiCompartmentModel) -> List[TissueParameter]:
    """
    Compiles a list of all tissue parameters present in the given multi-compartment model.

    :param tissue_model: A dmipy multi-compartment model.
    :return: A list with tissue parameters.
    """
    parameters = []
    for model, model_name in zip(tissue_model.models, tissue_model.model_names):
        for parameter_name in model.parameter_names:
            value = np.array(getattr(model, parameter_name), dtype=np.float64, copy=True)
            scale = model.parameter_scales[parameter_name]
            cardinality = model.parameter_cardinality[parameter_name]

            if value is None:
                raise ValueError(f'Parameter {parameter_name} of model {model_name} has no value.')

            # Iterate over tissue parameter vectors.
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


def get_dmipy_acquisition_scheme(acquisition_scheme: AcquisitionScheme) -> DmipyAcquisitionScheme:
    """
    Converts a MICROtool acquisition scheme into a dmipy acquisition scheme.

    :param acquisition_scheme: a MICROtool acquisition scheme.
    :return: a dmipy acquisition scheme.
    """
    b_values = acquisition_scheme.get_b_values() * 1e6  # Convert from s/mm² to s/m².
    b_vectors = acquisition_scheme.get_b_vectors()
    pulse_widths = acquisition_scheme.get_pulse_widths()
    pulse_intervals = acquisition_scheme.get_pulse_intervals()

    return acquisition_scheme_from_bvalues(b_values, b_vectors, pulse_widths, pulse_intervals)


class DmipyExperiment:
    def __init__(self, tissue_model: MultiCompartmentModel):
        self._tissue_model = tissue_model
        self._tissue_parameters = get_parameters(tissue_model)

        # Get the baseline parameter vector.
        self._parameter_baseline = np.array([parameter.value for parameter in self._tissue_parameters])

        # Calculate finite differences and corresponding parameter vectors for calculating derivatives.
        h = np.array([parameter.scale * 1e-6 for parameter in self._tissue_parameters])
        self._parameter_vectors = self._parameter_baseline + np.diag(h)
        self._reciprocal_h = (1 / h).reshape(-1, 1)

    def simulate(self, acquisition_scheme: Union[AcquisitionScheme, DmipyAcquisitionScheme]) -> np.ndarray:
        if isinstance(acquisition_scheme, AcquisitionScheme):
            acquisition_scheme = get_dmipy_acquisition_scheme(acquisition_scheme)

        # Evaluate the dmipy model.
        return self._tissue_model.simulate_signal(acquisition_scheme, self._parameter_baseline)

    def jacobian(self, acquisition_scheme: Union[AcquisitionScheme, DmipyAcquisitionScheme]) -> np.ndarray:
        """
        Calculate the M×N Jacobian matrix of the derivatives of the signal strength to the tissue parameter values,
        where M is the number of tissue parameters and N is the number of measurements.

        :param acquisition_scheme: A MICROtool acquisition scheme.
        :return: An M×N numpy array.
        """
        if isinstance(acquisition_scheme, AcquisitionScheme):
            acquisition_scheme = get_dmipy_acquisition_scheme(acquisition_scheme)

        # Evaluate the dmipy model on the baseline and on the parameter vectors with finite differences.
        baseline = self._tissue_model.simulate_signal(acquisition_scheme, self._parameter_baseline)
        differences = self._tissue_model.simulate_signal(acquisition_scheme, self._parameter_vectors) - baseline

        # Divide by the finite differences to obtain the derivatives.
        jac = (differences * self._reciprocal_h).T

        return jac

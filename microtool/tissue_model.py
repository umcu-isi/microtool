"""
The TissueModel is a base class for different kinds of tissue models (different types of models and/or different types
 of modelling-software).

In order to simulate the MR signal in response to a MICROtool acquisition scheme, the TissueModel first translates the
 acquisition scheme to a modelling-software specific one.
"""

from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from .acquisition_scheme import AcquisitionScheme, InversionRecoveryAcquisitionScheme
from .optimize import LossFunction, crlb_loss


@dataclass
class TissueParameter:
    # noinspection PyUnresolvedReferences
    """
    Defines a scalar tissue parameter and its properties.

    :param value: Parameter value.
    :param scale: The typical parameter value scale (order of magnitude).
    :param use: Specifies if the parameter should be included in the optimization (default = True).
    """
    value: float
    scale: float
    use: bool = True

    def __str__(self):
        return f'{self.value} (scale: {self.scale}, use: {self.use})'


class TissueModel:
    _parameters: Dict[str, TissueParameter]

    def __call__(self, scheme: AcquisitionScheme) -> np.ndarray:
        """
        Calculates the MR signal attenuation.

        :param scheme: An AcquisitionScheme.
        :return: A array with N signal attenuation values, where N is the number of samples.
        """
        raise NotImplementedError()

    def jacobian(self, scheme: AcquisitionScheme) -> np.ndarray:
        """
        Calculates the change in MR signal attenuation due to a change in the tissue model parameters.

        The order of the tissue parameters is the same as returned by get_parameters().

        :param scheme: An AcquisitionScheme.
        :return: An N×M Jacobian matrix, where N is the number of samples and M is the number of tissue parameters.
        """
        raise NotImplementedError()

    def optimize(
            self,
            scheme: AcquisitionScheme,
            noise_var: float,
            loss: LossFunction = crlb_loss,
            method: Optional[str] = None) -> OptimizeResult:
        """
        Optimizes the free parameters in the given MR acquisition scheme such that the loss is minimized.
        The loss function should be of type LossFunction, which takes an N×M Jacobian matrix, an array with M parameter
        scales, and the noise variance. The loss function should return a scalar loss. N is the number of measurements
        in the acquisition and M is the number of tissue parameters.

        :param scheme: The MR acquisition scheme to be optimized.
        :param noise_var: Noise variance on the MR signal attenuation.
        :param loss: a function of type LossFunction.
        :param method: Type of solver. See the documentation for scipy.optimize.minimize
        :return: A scipy.optimize.OptimizeResult object.
        """
        scales = [value.scale for value in self._parameters.values()]
        uses = [value.use for value in self._parameters.values()]
        acquisition_parameter_scales = scheme.get_free_parameter_scales()
        x0 = scheme.get_free_parameters() / acquisition_parameter_scales
        bounds = scheme.get_free_parameter_bounds()

        # Calculating the loss involves passing the new parameters to the acquisition scheme, calculating the tissue
        # model's Jacobian matrix and evaluating the loss function.
        def calc_loss(x: np.ndarray):
            scheme.set_free_parameters(x * acquisition_parameter_scales)
            jac = self.jacobian(scheme)
            return loss(jac, scales, uses, noise_var)

        result = minimize(calc_loss, x0, method=method, bounds=bounds)
        if 'x' in result:
            scheme.set_free_parameters(result['x'] * acquisition_parameter_scales)

        return result

    def get_parameters(self) -> Dict[str, TissueParameter]:
        return self._parameters

    def set_scale(self, name: str, scale: float):
        self._parameters[name].scale = scale

    def set_use(self, name: str, use: bool):
        self._parameters[name].use = use

    def __str__(self) -> str:
        parameter_str = '\n'.join(f'    - {key}: {value}' for key, value in self._parameters.items())
        return f'Tissue model with {len(self._parameters)} scalar parameters:\n{parameter_str}'


# TODO: Take T2* and relaxation parameter distributions into account. See eq. 5 and 6 in
#  https://www.ncbi.nlm.nih.gov/books/NBK567564/
@dataclass
class RelaxationTissueModel(TissueModel):
    # noinspection PyUnresolvedReferences
    """
    Defines a tissue by its relaxation parameters.

    Rather than varying TR to achieve different T1 weightings, Mulkern et al. (2000) incorporate an inversion pulse
    prior to the 90° pulse in the diffusion-weighted SE sequence for simultaneous D-T1 measurement.

    See section 7.4.2 of 'Advanced Diffusion Encoding Methods in MRI', Topgaard D, editor (2020):
    https://www.ncbi.nlm.nih.gov/books/NBK567564
    """
    def __init__(self, t1: float, t2: float, s0: float = 1.0):
        self._parameters = {
            't1': TissueParameter(value=s0, scale=t1, use=False),
            't2': TissueParameter(value=s0, scale=t2),
            's0': TissueParameter(value=s0, scale=s0, use=False),
        }

    # TODO: Support other relaxation-acquisition schemes, e.g. Union[SpinEchoAcquisitionScheme,
    #  InversionRecoveryAcquisitionScheme] and switch model based on these.
    def __call__(self, scheme: InversionRecoveryAcquisitionScheme) -> np.ndarray:
        ti = scheme.get_inversion_times()
        tr = scheme.get_repetition_times()
        te = scheme.get_echo_times()
        t1 = self._parameters['t1'].value
        t2 = self._parameters['t2'].value
        s0 = self._parameters['s0'].value

        ti_t1 = np.exp(-ti / t1)
        tr_t1 = np.exp(-tr / t1)
        te_t2 = np.exp(-te / t2)

        return s0 * (1 - 2 * ti_t1 + tr_t1) * te_t2

    def jacobian(self, scheme: InversionRecoveryAcquisitionScheme) -> np.ndarray:
        ti = scheme.get_inversion_times()
        tr = scheme.get_repetition_times()
        te = scheme.get_echo_times()
        t1 = self._parameters['t1'].value
        t2 = self._parameters['t2'].value
        s0 = self._parameters['s0'].value

        ti_t1 = np.exp(-ti / t1)
        tr_t1 = np.exp(-tr / t1)
        te_t2 = np.exp(-te / t2)

        # Calculate the derivative of the signal attenuation to T1, T2 and S0.
        return np.array([
            s0 * (-2 * ti * ti_t1 + tr * tr_t1) * te_t2 / (t1 ** 2),  # δS(S0, T1, T2) / δT1
            s0 * te * (1 - 2 * ti_t1 + tr_t1) * te_t2 / (t2 ** 2),  # δS(S0, T1, T2) / δT2
            (1 - 2 * ti_t1 + tr_t1) * te_t2,   # δS(S0, T1, T2) / δS0
        ]).T

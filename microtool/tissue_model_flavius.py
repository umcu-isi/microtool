import numpy as np

from microtool.acquisition_scheme_flavius import FlaviusAcquisitionScheme
from microtool.tissue_model import TissueModel, TissueParameter


class FlaviusSignalModel(TissueModel):
    def __init__(self, t2: float, diffusivity: float, s0: float = 1.0):
        """
        :param t2: The tissues T2 in [ms]
        :param diffusivity: The tissue diffusivity in [mm^2 / s]
        :param s0: signal at the zeroth measurement [dimensionless]
        """
        super().__init__({
            'T2': TissueParameter(value=t2, scale=t2),
            'Diffusivity': TissueParameter(value=diffusivity, scale=diffusivity),
            'S0': TissueParameter(value=s0, scale=s0, optimize=False),
        })

    def __call__(self, scheme: FlaviusAcquisitionScheme) -> np.ndarray:
        # The signal equation for this model S = S0*exp(-b.*D).*exp(-TE/T2)

        bvalues = scheme.b_values
        te = scheme.echo_times

        b_D = np.exp(-bvalues * self['Diffusivity'].value)
        te_t2 = np.exp(- te / self['T2'].value)
        return self['S0'].value * b_D * te_t2

    def jacobian(self, scheme: FlaviusAcquisitionScheme) -> np.ndarray:
        # Acquisition parameters
        bvalues = scheme.b_values
        te = scheme.echo_times

        # tissuemodel parameters
        D = self['Diffusivity'].value
        T2 = self['T2'].value
        S0 = self['S0'].value

        # Exponents
        b_D = np.exp(-bvalues * D)
        te_t2 = np.exp(- te / T2)

        jac = [te * S0 * b_D * te_t2 / T2 ** 2, - bvalues * S0 * b_D * te_t2, b_D * te_t2]
        return np.array(jac).T

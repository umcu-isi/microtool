from copy import deepcopy

from dmipy.core.modeling_framework import MultiCompartmentModel
from dmipy.data import saved_acquisition_schemes
from dmipy.signal_models.gaussian_models import G1Ball
from matplotlib import pyplot as plt

from microtool.dmipy import convert_dmipy_scheme2diffusion_scheme, DmipyTissueModel
from microtool.tissue_model import RelaxedMultiTissueModel

if __name__ == "__main__":
    ball = G1Ball(lambda_iso=2e-9)
    ball = MultiCompartmentModel([ball])
    ball = DmipyTissueModel(ball)
    multi_model = RelaxedMultiTissueModel([ball], [1.0], [100.0])

    dmipy_scheme = saved_acquisition_schemes.wu_minn_hcp_acquisition_scheme()
    scheme = convert_dmipy_scheme2diffusion_scheme(dmipy_scheme)

    multi_model['vf_0'].fit_flag = False
    print(scheme)
    print(multi_model)

    actual_signal = multi_model(scheme)
    plt.figure("Signal")
    plt.plot(actual_signal, '.')

    # MTM fit result
    mtm_result = multi_model.fit(scheme, actual_signal, method="trust-constr")
    print("MTM fit information:")
    mtm_result.print_fit_information()
    print('\n')

    fit_model = deepcopy(multi_model)
    fit_model.set_fit_parameters(mtm_result.fitted_parameters)

    multi_model.print_comparison(fit_model)

    plt.figure("residuals MTM")
    plt.plot(fit_model(scheme) - actual_signal, '.')

    plt.show()

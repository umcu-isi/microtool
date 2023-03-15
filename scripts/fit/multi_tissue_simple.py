import numpy as np
from matplotlib import pyplot as plt

from microtool.acquisition_scheme import EchoScheme
from microtool.tissue_model import ExponentialTissueModel, MultiTissueModel

if __name__ == "__main__":
    model = ExponentialTissueModel(T2=10.0)
    multi_model = MultiTissueModel([model], [1.0])
    scheme = EchoScheme(np.linspace(10, 20, num=30))

    actual_signal = multi_model(scheme)
    plt.figure("Signal")
    plt.plot(actual_signal, '.')

    # Curve fit result
    cf_result = model.fit(scheme, actual_signal)
    print("Curve fit result:", cf_result.fit_information, cf_result.fitted_parameters)
    plt.figure("Fitted signal prediction ETM")
    model.set_fit_parameters(cf_result.fitted_parameters)
    plt.plot(model(scheme) - actual_signal, '.')

    # MTM fit result
    mtm_result = multi_model.fit(scheme, actual_signal, method='DE')
    print("MTM fit result:", mtm_result.fit_information, mtm_result.fitted_parameters)
    plt.figure("Fitted signal prediction MTM")
    plt.plot(multi_model(scheme) - actual_signal, '.')

    plt.show()

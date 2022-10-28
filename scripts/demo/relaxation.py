import matplotlib.pyplot as plt
import numpy as np

from microtool import optimize, acquisition_scheme, tissue_model
from microtool.utils.plotting import LossInspector


def main():
    # # Inversion recovery

    # ## 1. Create a tissue model specifying a T1 and T2
    relaxation_model = tissue_model.RelaxationTissueModel(t1=900, t2=90)
    print(relaxation_model)

    # ## 2. Create an initial inversion-recovery acquisition scheme
    # Initial TR = 500 ms, initial TE = 10 ms, initial TI = {50, ..., 400} ms

    tr = np.array([500, 500, 500, 500, 500, 500, 500, 500])
    te = np.array([10, 10, 10, 10, 20, 20, 20, 20])
    ti = np.array([50, 100, 150, 200, 250, 300, 350, 400])

    ir_scheme = acquisition_scheme.InversionRecoveryAcquisitionScheme(tr, te, ti)
    print(ir_scheme)

    plt.figure(figsize=(6, 4))
    plt.plot(relaxation_model(ir_scheme), '.')
    plt.xlabel('Measurement')
    plt.ylabel('Signal attenuation')

    # ## 3. Optimize the acquisition scheme
    noise_variance = 0.02
    scheme_optimal, _ = optimize.optimize_scheme(ir_scheme, relaxation_model, noise_variance,
                                                 method='differential_evolution')

    print(scheme_optimal)
    plt.figure(figsize=(6, 4))
    plt.plot(relaxation_model(scheme_optimal), '.')
    plt.xlabel('Measurement')
    plt.ylabel('Signal attenuation')

    lossinspector = LossInspector(scheme_optimal, relaxation_model, noise_var=noise_variance)
    lossinspector.plot([{"InversionTime": 1}, {"RepetitionTimeExcitation": 2}])

    plt.show()


if __name__ == "__main__":
    main()

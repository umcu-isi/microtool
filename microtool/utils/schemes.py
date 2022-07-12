import numpy as np
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues, DmipyAcquisitionScheme
from .gradient_sampling import sample_uniform


def alexander2008() -> DmipyAcquisitionScheme:
    """
    This function does the setup for the acquisition scheme as defined by alexander et al.

    :return: Acquisition scheme as defined by Alexander etal.
    """

    # setting some random b0 measurements because alexander does not specify.
    n_b0 = 18
    b0 = np.zeros(n_b0)
    zero_directions = sample_uniform(n_b0)
    zero_delta = np.repeat(0.007, n_b0)
    zero_Delta = np.repeat(0.012, n_b0)

    n_measurements = 4
    n_directions = 30
    # Extending b_values array for every direction

    bvalues = np.repeat(np.array([17370, 3580, 1216, 1205]) * 1e6, n_directions)
    delta = np.repeat(np.array([0.019, 0.016, 0.007, 0.007]), n_directions)
    Delta = np.repeat(np.array([0.024, 0.027, 0.012, 0.012]), n_directions)
    gradient_directions = np.tile(sample_uniform(n_directions), (n_measurements, 1))

    # Prepending b0 measurements
    bvalues = np.concatenate([b0, bvalues])
    delta = np.concatenate([zero_delta, delta])
    Delta = np.concatenate([zero_Delta, Delta])
    gradient_directions = np.concatenate([zero_directions, gradient_directions], axis=0)

    return acquisition_scheme_from_bvalues(bvalues, gradient_directions, delta, Delta)

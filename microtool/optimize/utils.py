import numpy as np
from numba import njit


@njit
def cartesian_product(jac: np.ndarray):
    # number of parameters (we use N for tissue parameters and M for Measurements)
    M, N = jac.shape
    derivative_term = np.zeros((N, N, M))
    for i in range(N):
        for j in range(N):
            derivative_term[i, j, :] = jac[:, i] * jac[:, j]
    return derivative_term

"""
Module containing sampling scheme for shell organised b-vectors using electrostatic optimization based on the following
paper DOI:10.1002/mrm.24736
"""

from itertools import combinations
from typing import List

import numpy as np
from scipy.optimize import minimize

from ..gradient_sampling.utils import sample_sphere_vectors, get_unit_vec_constraint, normalize, total_potential


def sample_shells_electrostatic(n_shells: List[int]) -> List[np.ndarray]:
    """
    Samples shells according to the cost functions as defined in DOI: 10.1002/mrm.24736

    :param n_shells: A list containing the number of desired samples per shell
    :return: A list containing the sphere samples as seperate list entries
    """
    vec_init = list(map(sample_sphere_vectors, n_shells))
    x0 = np.concatenate(list(map(np.ravel, vec_init)))

    result = minimize(cost_scipy, x0, args=(n_shells, 0.5), constraints=get_unit_vec_constraint())
    return list(map(normalize, repack_shells(result['x'], n_shells)))


def cost_scipy(x: np.ndarray, n_shells: List[int], alpha: float = 0.5) -> float:
    """

    :param x: The flat array containing all the vectors
    :param n_shells: Number of vectors in each shell
    :param alpha: The relative weighting of inter and intra shell cost/electrostatic energy
    :return: The cost associated with this set of vectors
    """
    # folding the parameter vector
    vectors = repack_shells(x, n_shells)
    return alpha * cost_inter_shell(vectors) + (1.0 - alpha) * cost_intra_shell(vectors)


def repack_shells(x: np.ndarray, n_shells: List[int]) -> List[np.ndarray]:
    """
    :param x: A flattened numpy ndarray that you wish to reorganize in shell structure
    :param n_shells: A list containing the number of samples on each shell
    :return: The vectors of each shell stored as a seperate list entry
    """
    vectors = []
    idx = 0
    for n_q in n_shells:
        stride = n_q * 3
        vectors.append(x[idx:(idx + stride)].reshape(n_q, 3))
        idx += stride

    return vectors


def cost_intra_shell(vectors: List[np.ndarray]) -> float:
    """
    Computes the pairwise electrostatic energy between shell pairs and adds and scales them to penalize similar
    orientations on different shells.

    :param vectors: The list of vector sets of the different q shells
    :return: Cost associated with this configuration
    """

    # total number of samples
    n_total = float(sum(list(map(len, vectors))))  # float type setting

    # go over the unique pair combinations of shells s!=t
    cost = 0.
    for vec_shell_s, vec_shell_t in combinations(vectors, 2):
        # now that we have unique shell pair we go over
        for i in range(vec_shell_s.shape[0]):
            for j in range(vec_shell_t.shape[0]):
                cost += total_potential(np.vstack((vec_shell_s[i, :], vec_shell_t[j, :])))
    return cost * (1. / (n_total ** 2))


def cost_inter_shell(vectors: List[np.ndarray]) -> float:
    """
    Computes the electrostatic energy in each shell individually.

    :param vectors: The list of vectors where each list entry represent a different q-shell
    :return: Cost/Electrostatic energy
    """
    # Get electrostatic energy for individual shells
    n_q = len(vectors)  # number of shells
    cost = 0.0
    for i in range(n_q):
        # scale the cost per shell by the number of samples on the shell
        cost += (1.0 / (float(vectors[i].shape[0]) ** 2)) * total_potential(vectors[i])
    return cost

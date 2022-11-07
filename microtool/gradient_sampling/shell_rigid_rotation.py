"""
Module containing sampling scheme for shell organised b-vectors using rigid rotation optimization based on the following
paper DOI:10.1002/mrm.24717
"""
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from microtool.gradient_sampling.uniform import sample_uniform
from microtool.gradient_sampling.utils import plot_shells, total_potential


def test_rotation():
    """ Quick test to see if it works """
    samples = sample_shells_rotation([4, 16, 32])
    plot_shells(samples)
    plt.show()


def sample_shells_rotation(n_shells: List[int]) -> List[np.ndarray]:
    """

    :param n_shells:
    :return:
    """

    initial_vectors = list(map(sample_uniform, n_shells))
    # initial is small rotation on all shells to prevent starting in a maximum plato.
    x0 = np.random.rand(len(n_shells) - 1, 2) * 2 * np.pi * 1e-5

    # carrying out the optimization
    result = minimize(cost_scipy, x0, args=initial_vectors)

    # reshaping the angles
    angles = result['x'].reshape(-1, 2)

    return rotate_shells(initial_vectors, angles)


def cost_scipy(x: np.ndarray, vec_shells: List[np.ndarray]) -> float:
    """
    Should rotate, make a backprojection and then compute cost

    :param x: Flattend theta,phi in other words polar, azimuthal pairs
    :param vec_shells: The list defining the number of samples per shell.
    :return: total electrostatic cost
    """
    # We always fix the first shell, hence we have one less angle pair than number of shells
    angles = x.reshape(len(vec_shells) - 1, 2)

    vecs_rotated = rotate_shells(vec_shells, angles)
    # Back projecting the shells into one array and computing the electrostatic energy
    return float(total_potential(np.concatenate(vecs_rotated, axis=0)))


def rotate_shells(vec_shells: List[np.ndarray], angles: np.ndarray) -> List[np.ndarray]:
    """
    Rotates the shells by angles respectively.

    :param vec_shells: samples for every shell as list entry
    :param angles: a set of angles of shape (shells, 2)
    :return: List of vectors rotated by given set of angles.
    """
    # rotate all but first shell

    # rotated_shells = [rotate_vectors(vec_shells[1 + i], angles[i, :]) for i in range(len(vec_shells)-1)]
    rotated_shells = list(map(rotate_vectors, vec_shells[1::], angles))
    # returning the full effect of rotating w.r.t first shell
    return [vec_shells[0], *rotated_shells]


def rotate_vectors(vectors: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """

    :param vectors: A set of vectors to be rotated, shape (n_vecs, 3)
    :param angles: A polar and azimuthal angle to rotate by (theta, phi), shape (2,)
    :return:
    """
    # Rotating by theta along the y-axis composed with rotating by phi along the z-axis
    rotation = Rotation.from_euler('yz', angles)
    return rotation.apply(vectors)

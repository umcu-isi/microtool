"""
Module for helper functions in gradient_sampling.
"""

from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance


def total_potential(vectors: np.ndarray) -> np.ndarray:
    """
    Computes the potential energy generated between pairs of points.

    :param vectors: an array of shape (n_vec,3)
    :return: pairwise potential array
    """
    return np.sum(1 / distance.pdist(vectors))


def normalize(vecs: np.ndarray) -> np.ndarray:
    """
    Normalizes a set of vectors.

    :param vecs: An array of shape (n_vecs, 3)
    :return: Vector set of shape (n_vecs, 3) with each vector length being unity
    """
    norms = np.linalg.norm(vecs, axis=1)
    result = np.zeros(vecs.shape)
    for i, norm in enumerate(norms):
        result[i] = vecs[i, :] / norm
    return result


def get_unit_vec_constraint() -> dict:
    """
    :return: Constraints for keeping vectors of length unity
    """
    return {'type': 'eq', 'fun': deviation_from_unit_vec}


def get_positive_half_spere_constraint() -> dict:
    return {"type": "ineq", "fun": distance_from_xy_plane}


def distance_from_xy_plane(x: np.ndarray) -> np.ndarray:
    """
    Simply return the z coordinates of the vectors
    :param x: The scipy array used during optimization
    :return: the array of z coordinates
    """
    vecs = get_vecs_from_scipy_array(x)
    return vecs[:, 2]


def deviation_from_unit_vec(x: np.ndarray) -> np.ndarray:
    """
    Computes the difference of the vector lengths from unity.

    The constraint function according to scipy standards. Meaning f(x) = 0 for equality type and f(x) <= 0 for
    inequality type. (output is allowed to be a 1D array )

    :param x: Parameter vector
    :return: array with difference from unit length for every vector
    """
    vecs = get_vecs_from_scipy_array(x)
    norms = np.linalg.norm(vecs, axis=1)
    return np.abs(norms - np.ones(norms.shape))


def get_vecs_from_scipy_array(x):
    vecs = np.reshape(x, (-1, 3))
    return vecs


def make_sphere(r):
    """
    Quick function for generating some points on a sphere for plotting purposes.
    """
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0 * pi:100j]
    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)
    return x, y, z


def plot_shells(shells: List[np.ndarray]) -> Tuple[plt.Figure, plt.Axes]:
    """
    Helper function for plotting q-shell configurations
    :param shells: List of vector collections for each shell
    :return: None, makes matplotlib figure. Display using plt.show()
    """
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(shells))))
    for i, vecs in enumerate(shells):
        c = next(color)
        scale_factor = float(i + 1)
        ax.scatter3D(scale_factor * vecs[:, 0], scale_factor * vecs[:, 1], scale_factor * vecs[:, 2], color=c)
        ax.plot_surface(*make_sphere(i + 1), alpha=0.2 / (i + 1), color='gray')

    plt.tight_layout()
    return fig, ax


def plot_shells_projected(shells: List[np.ndarray]) -> None:
    plt.figure()
    ax = plt.axes(projection='3d')

    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(shells))))
    for i, vecs in enumerate(shells):
        c = next(color)
        scale_factor = float(i + 1)
        ax.scatter3D(vecs[:, 0], vecs[:, 1], vecs[:, 2], color=c, label=f'shell {i + 1}')

    ax.legend()
    ax.plot_surface(*make_sphere(1), alpha=0.2, color='gray')
    plt.tight_layout()


def plot_vectors(vectors: np.ndarray, title: str) -> None:
    """
    Helper function for plotting a set of vectors
    :param vectors: (num, 3) shaped array of vectors
    :param title: Title for the plot
    :return: None, display using plt.show()
    """
    plt.figure(title)
    ax = plt.axes(projection='3d')
    plt.title(title)
    ax.scatter3D(vectors[:, 0], vectors[:, 1], vectors[:, 2])
    plt.tight_layout()


def unitvector_to_angles(vector: np.ndarray) -> np.ndarray:
    """ Gives the polar and azimuthal angle of a unit vector in domains [0,pi] and [-pi,+pi] respectively

    :param vector: A unit vector in R^3 or array of shape (n_vectors,3)
    :return: array of shape (n_angles, 2)
    :raises: ValueError if not unit vector or incorrect dimensions
    """
    if len(vector.shape) != 2:
        raise ValueError("Invalid unit vector collection, expecting shape (n_vectors,3).")
    if vector.shape[1] != 3:
        raise ValueError("Invalid shape, expecting the unit vectors in axis 1, i.e., vector.shape[1] == 3")
    if not np.all(np.isclose(np.linalg.norm(vector, axis=1), np.ones(vector.shape[0]))):
        raise ValueError("The collection contains at least one non unit vector.")

    x, y, z = (vector[:, i] for i in range(3))
    # The polar angle theta (arccos maps into [0,pi])
    theta = np.arccos(z)

    # Tha azimuthal angle phi (note that arctan2 maps into [-pi, + pi] as dmipy expects)
    phi = np.arctan2(y, x)
    return np.stack([theta, phi], axis=-1)


def angles_to_unitvectors(angles: np.ndarray) -> np.ndarray:
    """
    Returns unit vector given an array of spherical angles
    :param angles: array in the form theta, phi; polar, azimuthal, shape (n_angles, 2)
    :return: unit vectors on the sphere of shape (n_vectors, 3)
    :raises: Value error if the array is not of two angles or incorrect shape
    """
    if len(angles.shape) != 2:
        raise ValueError("Incorrect array shape")
    if angles.shape[1] != 2:
        raise ValueError("Expecting angles in the axis 1")

    theta = angles[:, 0]
    phi = angles[:, 1]
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(theta)
    return np.stack([x, y, z], axis=-1)


def sample_sphere_vectors(ns: int = 100) -> np.ndarray:
    return angles_to_unitvectors(sample_sphere_angles(ns))


def sample_sphere_angles(ns: int = 100) -> np.ndarray:
    """ Makes uniform RANDOM samples on the sphere in terms of spherical angles

    :param ns: Number of samples you wish to take on the sphere
    :return: The spherical angles in shape (number of samples, 2)
    """
    # TODO: instead of random sampling points, we could use a spherical Fibonacci lattice point set. E.g.:
    # i = np.arange(0, n, dtype=float) + 0.5
    # phi = np.arccos(1 - i / n)  # Use 1 - 2*i / n to sample a full sphere.
    # golden_ratio = 0.5 * (1 + np.sqrt(5))
    # theta = 2 * np.pi * i / golden_ratio
    # x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)

    # sample the uniform distribution for both angles
    urng = np.random.default_rng()
    mu = urng.random(size=(ns, 2))

    # map first element into [0,pi]
    mu[:, 0] = mu[:, 0] * np.pi

    # First shift then scale to go to [-pi,+pi] as domain
    mu[:, 1] = mu[:, 1] * 2 * np.pi - np.pi
    return mu

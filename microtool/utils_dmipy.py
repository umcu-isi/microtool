from math import isclose
from typing import Tuple

import numpy as np


def eigenvector_to_angles(vector: np.ndarray) -> np.ndarray:
    """ Gives the polar and azimuthal angle of a unit vector in domains [0,pi] and [-pi,+pi] respectively

    :param vector: A unit vector in R^3
    :return: tuple of form polar angle, azimuthal angle
    :raises: ValueError if not unit vector or incorrect dimensions
    """
    if not len(vector) == 3:
        raise ValueError("Invalid unit vector dimensions")
    if not isclose(np.linalg.norm(vector), 1):
        raise ValueError("Not a unit vector")

    x, y, z = vector
    # The polar angle theta (arccos maps into [0,pi])
    theta = np.arccos(z)

    # Tha azimuthal angle phi (note that arctan2 maps into [-pi, + pi] as dmipy expects
    phi = np.arctan2(y / x)
    return np.array([theta, phi])


def angles_to_eigenvectors(angles: np.ndarray) -> np.ndarray:
    """
    Returns unit vector given an array of spherical angles
    :param angles: array in the form theta, phi; polar, azimuthal
    :return: unit vector on the sphere
    :raises: Value error if the array is not of two angles
    """
    theta = angles[:,0]
    phi = angles[:,1]
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(theta)
    return np.stack([x, y, z], axis=-1)


def sample_sphere(ns: int = 1) -> np.ndarray:
    """ Makes uniform RANDOM samples on the sphere in terms of spherical angles

    :param ns: Number of samples you wish to take on the sphere
    :return: The spherical angles in shape (2, number of samples)
    """
    # sample the uniform distribution for both angles
    urng = np.random.default_rng()
    mu = urng.random(size=(ns, 2))

    # map first element into [0,pi]
    mu[:, 0] = mu[:, 0] * np.pi

    # First shift then scale to go to [-pi,+pi] as domain
    mu[:, 1] = mu[:, 1] * 2 * np.pi - np.pi
    return mu

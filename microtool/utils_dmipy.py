import numpy as np


def eigenvector_to_angles(vector: np.ndarray) -> np.ndarray:
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
    phi = np.arctan2(y / x)
    return np.stack([theta, phi], axis=-1)


def angles_to_eigenvectors(angles: np.ndarray) -> np.ndarray:
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


def sample_sphere(ns: int) -> np.ndarray:
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

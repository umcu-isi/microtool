import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance
from scipy.optimize import minimize, NonlinearConstraint


def main():
    """ Demonstrating the use of these functions"""
    # Uniform sphere sampling
    angle_samples = sample_sphere()
    vector_samples = angles_to_eigenvectors(angle_samples)
    _plot_vectors(vector_samples, "Using sample_sphere")

    # Optimizing the coulomb potential constraining the points to the sphere
    samples = sample_uniform()
    _plot_vectors(samples, "Electrostatic optimization")
    plt.show()


def sample_uniform(ns: int = 100) -> np.ndarray:
    """
    Using electrostatic energy minimization we generate uniform samples on the sphere

    :param ns: Number of samples on the sphere
    :return: the unit vectors uniform on sphere in shape (ns,3)
    """
    vec_init = angles_to_eigenvectors(sample_sphere(ns))
    x0 = vec_init.flatten()
    constraints = NonlinearConstraint(get_norms, make_bounds(x0), make_bounds(x0))
    result = minimize(cost_fun, x0, constraints=constraints)
    return result['x'].reshape((-1, 3))


def make_bounds(x):
    vecs = np.reshape(x, (-1, 3))
    return np.ones(vecs.shape[0])


def get_norms(x):
    """
    Helper function for constraining the vectors to unit length (i.e. on the sphere)
    :param x:
    :return:
    """
    vecs = np.reshape(x, (-1, 3))
    return np.linalg.norm(vecs, axis=1)


def cost_fun(x):
    """
    The cost function that models the energy of point charges
    :param x: array of shape (ns * 3, ) , (just the flattend unit vector samples)
    :return: scalar representing the total energy of the system
    """
    vecs = np.reshape(x, (-1, 3))
    return np.sum(potential(vecs))


def potential(vectors: np.ndarray) -> np.ndarray:
    """
    Computes the potential energy generated between pairs of points
    :param vectors: an array of shape (n_vec,3)
    :return: pairwise potential array
    """
    return 1 / distance.pdist(vectors)


def _plot_vectors(vectors: np.ndarray, title: str) -> None:
    """
    Helper function for plotting a set of vectors
    :param vectors: (num, 3) shaped array of vectors
    :param title: Title for the plot
    :return: None
    """
    plt.figure(title)
    ax = plt.axes(projection='3d')
    plt.title(title)
    ax.scatter3D(vectors[:, 0], vectors[:, 1], vectors[:, 2])
    plt.tight_layout()


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


def sample_sphere(ns: int = 100) -> np.ndarray:
    """ Makes uniform RANDOM samples on the sphere in terms of spherical angles

    :param ns: Number of samples you wish to take on the sphere
    :return: The spherical angles in shape (number of samples, 2)
    """
    # sample the uniform distribution for both angles
    urng = np.random.default_rng()
    mu = urng.random(size=(ns, 2))

    # map first element into [0,pi]
    mu[:, 0] = mu[:, 0] * np.pi

    # First shift then scale to go to [-pi,+pi] as domain
    mu[:, 1] = mu[:, 1] * 2 * np.pi - np.pi
    return mu


if __name__ == "__main__":
    main()

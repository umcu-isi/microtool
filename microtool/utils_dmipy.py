from itertools import combinations
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import distance


def test_uniform():
    # Uniform sphere sampling
    vector_samples = sample_sphere_vectors()
    _plot_vectors(vector_samples, "Using sample_sphere")

    # Optimizing the coulomb potential constraining the points to the sphere
    samples = sample_uniform()
    print(np.linalg.norm(samples, axis=1))
    _plot_vectors(samples, "Electrostatic optimization")
    plt.show()


def test_shells():
    result = sample_q_shells([9, 9, 9])
    _plot_shells(result)
    plt.show()


def main():
    """ Demonstrating the use of these functions"""
    result = sample_q_shells([9, 9, 9])
    _plot_shells(result)
    plt.show()


def sample_q_shells(n_shells: List[int]) -> List[np.ndarray]:
    vec_init = list(map(sample_sphere_vectors, n_shells))
    x0 = np.concatenate(list(map(np.ravel, vec_init)))

    result = minimize(cost_scipy, x0, args=(n_shells, 0.5), constraints=get_constraints())
    return list(map(normalize, repack_shells(result['x'], n_shells)))


def cost_scipy(x: np.ndarray, n_shells: List[int], alpha: float = 0.5) -> float:
    # folding the parameter vector
    vectors = repack_shells(x, n_shells)
    return alpha * cost_inter_shell(vectors) + (1.0 - alpha) * cost_intra_shell(vectors)


def repack_shells(x, n_shells) -> List[np.ndarray]:
    vectors = []
    idx = 0
    for n_q in n_shells:
        stride = n_q * 3
        vectors.append(x[idx:(idx + stride)].reshape(n_q, 3))
        idx += stride

    return vectors


def cost_intra_shell(vectors: List[np.ndarray]) -> float:
    # total number of samples
    n_total = float(sum(list(map(len, vectors))))  # float type setting

    # go over the unique pair combinations of shells s!=t
    cost = 0.
    for vec_shell_s, vec_shell_t in combinations(vectors, 2):
        for i in range(vec_shell_s.shape[0]):
            for j in range(vec_shell_t.shape[0]):
                cost += total_potential(np.vstack((vec_shell_s[i, :], vec_shell_t[j, :])))
    return cost * (1. / (n_total ** 2))


def cost_inter_shell(vectors: List[np.ndarray]) -> float:
    # Get electrostatic energy for individual shells
    n_q = len(vectors)  # number of shells
    cost = 0.0
    for i in range(n_q):
        # scale the cost per shell by the number of samples on the shell
        cost += (1.0 / (float(vectors[i].shape[0]) ** 2)) * total_potential(vectors[i])
    return cost


def sample_uniform(ns: int = 100) -> np.ndarray:
    """
    Using electrostatic energy minimization we generate uniform samples on the sphere

    :param ns: Number of samples on the sphere
    :return: the unit vectors uniform on sphere in shape (ns,3)
    """
    vec_init = sample_sphere_vectors(ns)
    x0 = vec_init.flatten()
    result = minimize(cost_fun, x0, constraints=get_constraints())
    return normalize(result['x'].reshape((-1, 3)))


def normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1)
    result = np.zeros(vecs.shape)
    for i, norm in enumerate(norms):
        result[i] = vecs[i, :] / norm
    return result


def get_constraints() -> dict:
    return {'type': 'eq', 'fun': constraint_function}


def constraint_function(x: np.ndarray) -> np.ndarray:
    vecs = np.reshape(x, (-1, 3))
    norms = np.linalg.norm(vecs, axis=1)
    return np.abs(norms - np.ones(norms.shape))


def cost_fun(x):
    """
    The cost function that models the energy of point charges
    :param x: array of shape (ns * 3, ) , (just the flattend unit vector samples)
    :return: scalar representing the total energy of the system
    """
    vecs = np.reshape(x, (-1, 3))
    return total_potential(vecs)


def total_potential(vectors: np.ndarray) -> np.ndarray:
    """
    Computes the potential energy generated between pairs of points
    :param vectors: an array of shape (n_vec,3)
    :return: pairwise potential array
    """
    return np.sum(1 / distance.pdist(vectors))


def make_sphere(r):
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0 * pi:100j]
    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)
    return x, y, z


def _plot_shells(shells: List[np.ndarray], title: str = None) -> None:
    plt.figure()
    ax = plt.axes(projection='3d')

    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(shells))))
    for i, vecs in enumerate(shells):
        c = next(color)
        scale_factor = float(i + 1)
        ax.scatter3D(scale_factor * vecs[:, 0], scale_factor * vecs[:, 1], scale_factor * vecs[:, 2], color=c)
        ax.plot_surface(*make_sphere(i + 1), alpha=0.2, color='gray')

    plt.tight_layout()


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


def sample_sphere_vectors(ns: int = 100) -> np.ndarray:
    return angles_to_eigenvectors(sample_sphere_angles(ns))


def sample_sphere_angles(ns: int = 100) -> np.ndarray:
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

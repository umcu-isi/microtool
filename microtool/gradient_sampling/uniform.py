"""
Module for uniform electrostatic sampling on the sphere.
"""

import pathlib

import numpy as np
from scipy.optimize import minimize

from microtool.gradient_sampling.utils import sample_sphere_vectors, get_unit_vec_constraint, normalize, \
    total_potential, get_positive_half_spere_constraint

module_folder = pathlib.Path(__file__).resolve().parent
folder = module_folder / 'gradient_directions'
folder.mkdir(exist_ok=True)


def sample_uniform_half_sphere(N: int) -> np.ndarray:
    """
    Calculates uniformly spaced points on the positive (z>0) half-sphere by generating 2N vectors on the full sphere and
    discarding vectors with z<0.

    :param N: The number of sampled unit vectors required
    :return: unit vectors in cartesian coordinates in shape (N,3) where all vectors are on the positive halfsphere
    """
    check_requested_vectors(N)
    base_name = "half_sphere_samples_"

    constraints = [get_unit_vec_constraint(), get_positive_half_spere_constraint()]

    return find_or_optimize(N, base_name, constraints)


def sample_uniform(ns: int = 100) -> np.ndarray:
    """
    Using electrostatic energy minimization we generate uniform samples on the sphere. We keep a lookup table for the
    samples meaning that if this sample has been previously computed we can load it from the disk.

    :param ns: Number of samples on the sphere
    :return: the unit vectors uniform on sphere in shape (ns,3)
    """
    check_requested_vectors(ns)
    base_name = "uniform_samples_"

    return find_or_optimize(ns, base_name, get_unit_vec_constraint())


def check_requested_vectors(ns):
    if ns == 0:
        raise ValueError("Requested number of vectors is 0?")


def find_or_optimize(ns: int, base_name, constraints):
    """
    Looks up if a vector collection was already optimzed, if not optimizes.
    """
    stored_samples = [sample_path.name for sample_path in list(folder.glob(base_name + '*'))]

    sample_name = base_name + str(ns) + '.npy'
    if sample_name in stored_samples:
        vectors = np.load(str(folder / sample_name))
    else:
        # if look up was unsuccessful we sample and write to disk

        vec_init = sample_sphere_vectors(ns)
        x0 = vec_init.flatten()
        result = minimize(cost_fun, x0, constraints=constraints)
        vectors = normalize(result['x'].reshape((-1, 3)))
        np.save(str(folder / sample_name), vectors)
    return vectors


def cost_fun(x):
    """
    The cost function that models the energy of point charges
    :param x: array of shape (ns * 3, ) , (just the flattend unit vector samples)
    :return: scalar representing the total energy of the system
    """
    vecs = np.reshape(x, (-1, 3))
    return total_potential(vecs)

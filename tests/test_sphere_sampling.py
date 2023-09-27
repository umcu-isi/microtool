import numpy as np
from matplotlib import pyplot as plt

from microtool.gradient_sampling import sample_uniform_half_sphere, sample_uniform
from microtool.gradient_sampling.utils import sample_sphere_vectors, plot_vectors


def test_sample_half_sphere():
    """
    Testing that number of requested vectors matches output
    """
    for n in range(2, 100):
        vecs = sample_uniform_half_sphere(n)
        assert vecs.shape[0] == n


def test_uniform():
    # Random samples on the sphere
    vector_samples = sample_sphere_vectors()
    plot_vectors(vector_samples, "Using sample_sphere")

    # Optimizing the coulomb potential constraining the points to the sphere
    samples = sample_uniform(100)
    print(np.linalg.norm(samples, axis=1))
    plot_vectors(samples, "Electrostatic optimization")
    plt.show()

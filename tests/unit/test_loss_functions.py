import numpy as np

from microtool.optimize.loss_functions import fisher_information_gauss, fisher_information_rice, gauss_loss, \
    rice_loss, ILL_LOSS


def test_fisher_information_gauss():
    d = 3.
    jac = np.array([[d]])  # d:1 relation between signal and variable. The information metric scales with d².
    noise_var = 5

    # With Gaussian noise, the amount of information that the signal carries is the inverse of the noise variance:
    fi = fisher_information_gauss(jac, np.array([]), noise_var)
    assert np.allclose(fi, d**2 / noise_var)


def test_fisher_information_rice():
    d = 3.
    jac = np.array([[d]])  # d:1 relation between signal and variable. The information metric scales with d².
    noise_var = 5

    # Because the Rician PDF depends on the signal amplitude, the Fisher information also does.

    # It is zero when the amplitude is zero.
    fi = fisher_information_rice(jac, np.array([0.]), noise_var)
    assert np.allclose(fi, 0)

    # The Rician PDF approximates a Gaussian PDF for high SNR, i.e. large amplitudes relative to the noise variance.
    snr = 50
    fi = fisher_information_rice(jac, np.array([np.sqrt(snr * noise_var)]), noise_var)
    assert np.allclose(fi, d**2 / noise_var, rtol=0.05)  # TODO: rtol could possibly be lowered?

    # TODO: fisher_information_rice uses an approximation that should deviate 4% at most. Test it using numerical
    #  integration.


def test_crlb():
    n = 10
    d = np.random.randn(n)
    jac = np.diag(d)  # d:1 relation between N measurements and variables without any covariance.
    noise_var = 0.3

    # With Gaussian noise, the parameter noise scales with the measurement noise.
    crlb = gauss_loss(jac, np.array([]), noise_var)
    assert np.allclose(crlb, np.sum(noise_var / d**2))

    # An ill-conditioned problem should return a high loss. Test with more parameters than measurements.
    n = 2
    m = 3
    jac = np.random.randn(n, m)
    signal = np.random.randn(n)
    for loss in (gauss_loss, rice_loss):
        crlb = loss(np.array(jac), signal, noise_var)
        assert np.allclose(crlb, ILL_LOSS)


def test_eigenvalue_versions():
    n = 10
    m = 5
    jac = np.random.randn(n, m)
    signal = np.random.randn(n)
    noise_var = 0.3

    # gauss_loss() uses eigenvalues to calculate the sum of lower bounds, which should match the sum of
    # gauss_loss.crlb, which uses matrix inversion.
    for loss in (gauss_loss, rice_loss):
        crlb_eig = loss(jac, signal, noise_var)
        crlb_inv = loss.crlb(jac, signal, noise_var).sum()
        assert np.allclose(crlb_eig, crlb_inv)

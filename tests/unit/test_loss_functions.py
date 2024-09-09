import numpy as np

from microtool.optimize.loss_functions import fisher_information_gauss, fisher_information_rice


def test_fisher_information_gauss():
    d = 3.
    jac = np.array([[d]])  # d:1 relation between signal and variable. The information metric scales with d².
    noise_var = 5

    # With Gaussian noise, the amount of information that the signal carries is the inverse of the noise variance:
    fi = fisher_information_gauss(jac, np.array([]), noise_var)
    assert np.allclose(fi, d * d / noise_var)


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
    assert np.allclose(fi, d * d / noise_var, rtol=0.05)  # TODO: rtol could possibly be lowered?

    # TODO: fisher_information_rice uses an approximation that should deviate 4% at most. Test it using numerical
    #  integration.

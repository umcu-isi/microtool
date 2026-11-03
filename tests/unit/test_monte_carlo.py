import numpy as np
from scipy import stats

from microtool.monte_carlo.simulation import MonteCarloSimulation


class _FitResult:
    def __init__(self, value):
        self.fitted_parameters = {"value": value}


class _RecordingModel:
    def __init__(self):
        self.fit_signals = []

    def __call__(self, scheme):
        return np.ones(3)

    def fit(self, scheme, signal, **fit_options):
        self.fit_signals.append(np.array(signal))
        return _FitResult(float(np.sum(signal)))


def test_monte_carlo_seed_repeats_noise_draws_across_models():
    first_model = _RecordingModel()
    second_model = _RecordingModel()
    noise = stats.norm(loc=0, scale=0.02)

    first = MonteCarloSimulation(
        object(), first_model, noise, n_sim=4, random_state=23
    )
    second = MonteCarloSimulation(
        object(), second_model, noise, n_sim=4, random_state=23
    )

    first_result = first.run()
    second_result = second.run()

    for first_signal, second_signal in zip(
        first_model.fit_signals, second_model.fit_signals
    ):
        np.testing.assert_array_equal(first_signal, second_signal)
    np.testing.assert_array_equal(first_result, second_result)

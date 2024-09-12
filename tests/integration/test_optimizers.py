# Test compatibility with SOMA and all scipy optimizers.
from microtool.acquisition_scheme import EchoScheme
from microtool.optimize import optimize_scheme
from microtool.optimize.loss_functions import compute_loss, gauss_loss
from microtool.tissue_model import ExponentialTissueModel


# List of optimizers available in SciPy 1.10
optimizers = [
    'differential_evolution',  # Calls scipy.optimize.differential_evolution
    'Nelder-Mead',
    'Powell',
    'CG',
    # 'BFGS',  # Requires a Jacobian
    # 'Newton-CG',  # Requires a Jacobian
    'L-BFGS-B',
    'TNC',
    'COBYLA',
    'SLSQP',
    'trust-constr',
    # 'dogleg',  # Requires a Jacobian
    # 'trust-ncg',  # Requires a Jacobian
    # 'trust-exact',  # Requires a Jacobian
    # 'trust-krylov',  # Requires a Jacobian
]


def test_optimize():
    # Define a very simple model and acquisition scheme.
    model = ExponentialTissueModel(t2=20)  # T2 = 20 ms
    scheme = EchoScheme(te=[60, 80, 100])  # TE = 60, 80, 100 ms
    noise_variance = 1.0

    # Initial loss.
    initial_loss = compute_loss(scheme, model, noise_variance, gauss_loss)

    # Try each optimizer with a single iteration and, as a sanity check, check if the loss did not increase.
    for method in optimizers:
        _, loss = optimize_scheme(scheme, model, noise_variance, loss=gauss_loss, method=method,
                                  solver_options={"maxiter": 1})
        assert loss <= initial_loss

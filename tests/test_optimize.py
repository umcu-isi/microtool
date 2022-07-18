"""
This module defines all pytests for the microtool.optimize and microtool.TissueModel.optimize functionality. We start by
defining unittests and increase complexity to integration tests.
"""

from copy import copy

import pytest
import numpy as np

from microtool import tissue_model, optimization_methods
from microtool.optimize import compute_loss, optimize_scheme
from microtool.utils import saved_schemes

# --------- Loading optimization methods
# scipy bound constrained and constrained optimization methods
scipy_methods = ['Nelder-Mead', 'L-BFGS-B', 'Powell', 'TNC', 'COBYLA', "SLSQP", 'trust-constr']

# Instantiating Optimizers with default control parameters for testing (excluding bruteforce since its slow)
custom_methods = [optimizer() for optimizer in optimization_methods.Optimizer.__subclasses__() if optimizer!=optimization_methods.BruteForce]
METHODS = [*scipy_methods, *custom_methods]

# --------- Basic tissuemodel for testing.

# The global noise, might parameterize in the future
NOISE = 0.02
# Simplest model
RELAXATION_MODEL = tissue_model.RelaxationTissueModel(t1=900, t2=90)


@pytest.mark.parametrize("optimization_method", METHODS)
@pytest.mark.parametrize(
    "scheme_factory", [saved_schemes.ir_scheme_increasing_parameters, saved_schemes.ir_scheme_repeated_parameters]
)
def test_optimizers(scheme_factory, optimization_method):
    """
    Testing if optimize_scheme actually reduces the loss
    """
    # copying to prevent test interferences
    model = copy(RELAXATION_MODEL)

    # For now we test with 3 pulses for time efficiency
    schemes = list(map(scheme_factory, [3, 4, 5, 6]))

    loss_non_optimal = [compute_loss(scheme, model, NOISE) for scheme in schemes]

    # If all initial loss values are non optimal we should raise an error message!
    if (np.array(loss_non_optimal) >= 1e9).all():
        with pytest.raises(ValueError):
            best_scheme, _ = optimize_scheme(schemes, model, NOISE, method=optimization_method, repeat=2)
    else:
        best_scheme, _ = optimize_scheme(schemes, model, NOISE, method=optimization_method, repeat=2)

        # We check if the best scheme was an improvement over its initial setting and this defines the succes of the optimizer;
        best_index = schemes.index(best_scheme)
        loss_optimal = [compute_loss(scheme, model, NOISE) for scheme in schemes]

        assert loss_optimal[best_index] < loss_non_optimal[best_index]

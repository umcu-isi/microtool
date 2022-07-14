"""
This module defines all pytests for the microtool.optimize and microtool.TissueModel.optimize functionality. We start by
defining unittests and increase complexity to integration tests.
"""

from copy import copy

import pytest

from microtool import tissue_model, optimize, optimization_methods
from microtool.utils import schemes

# --------- Loading optimization methods
# scipy bound constrained and constrained optimization methods
scipy_methods = ['Nelder-Mead', 'L-BFGS-B', 'Powell', 'TNC', 'COBYLA', "SLSQP", 'trust-constr']

# Instantiating Optimizers with default control parameters for testing
custom_methods = [optimizer() for optimizer in optimization_methods.Optimizer.__subclasses__()]
METHODS = scipy_methods + custom_methods

# --------- Basic tissuemodel for testing.

# The global noise, might parameterize in the future
NOISE = 0.02
# Simplest model
RELAXATION_MODEL = tissue_model.RelaxationTissueModel(t1=900, t2=90)


@pytest.mark.parametrize("optimization_method", METHODS)
@pytest.mark.parametrize(
    "scheme_factory", [schemes.ir_scheme_increasing_parameters, schemes.ir_scheme_repeated_parameters]
)
def test_optimizers(scheme_factory, optimization_method):
    """
    Testing if optimize_scheme actually reduces the loss
    """
    # copying to prevent test interferences
    model = copy(RELAXATION_MODEL)

    # For now we test with 3 pulses for time efficiency
    scheme = scheme_factory(n_pulses=3)

    loss_non_optimal = optimize.compute_loss(model, scheme, NOISE, optimize.crlb_loss)

    result = optimize.optimize_scheme(scheme, model, NOISE, method=optimization_method)
    optimized_parameters = result['x'] * scheme.free_parameter_scales
    loss_optimal = optimize.compute_loss(model, scheme, NOISE, optimize.crlb_loss)
    assert loss_optimal < loss_non_optimal

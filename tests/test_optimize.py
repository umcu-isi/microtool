"""
This module defines all pytests for the microtool.optimize and microtool.TissueModel.optimize functionality. We start by
defining unittests and increase complexity to integration tests.
"""

import pytest

import numpy as np
from microtool import tissue_model, acquisition_scheme, optimize
from microtool.utils import schemes
from copy import copy

# The global noise, might parameterize in the future
NOISE = 0.02

# Simplest model
RELAXATION_MODEL = tissue_model.RelaxationTissueModel(t1=900, t2=90)
# Simplest scheme
tr = np.array([500, 500, 500, 500, 500, 500, 500, 500])
te = np.array([10, 10, 10, 10, 20, 20, 20, 20])
ti = np.array([50, 100, 150, 200, 250, 300, 350, 400])
IR_SCHEME = acquisition_scheme.InversionRecoveryAcquisitionScheme(tr, te, ti)


# TODO: Parameterize test with different models and optimizers in the future

@pytest.mark.parametrize(
    "scheme_factory", [schemes.ir_scheme_increasing_parameters, schemes.ir_scheme_repeated_parameters]
)
def test_relaxation(scheme_factory):
    """
    Testing if tissuemodel.optimize actually reduces the loss
    """
    # copying to prevent test interferences
    model = copy(RELAXATION_MODEL)

    # For now we test with 10 pulses for time efficiency
    scheme = scheme_factory(n_pulses = 10)

    loss_non_optimal = optimize.compute_loss(model, scheme, NOISE, optimize.crlb_loss)

    result = optimize.optimize_scheme(scheme, model, NOISE)
    optimized_parameters = result['x'] * scheme.free_parameter_scales
    loss_optimal = optimize.compute_loss(model, scheme, NOISE, optimize.crlb_loss)
    assert loss_optimal < loss_non_optimal

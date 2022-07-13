"""
This module defines all pytests for the microtool.optimize and microtool.TissueModel.optimize functionality. We start by
defining unittests and increase complexity to integration tests.
"""

import pytest

import numpy as np
from microtool import tissue_model, acquisition_scheme, optimize
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


def make_scheme(n_pulses: int) -> acquisition_scheme.InversionRecoveryAcquisitionScheme:
    """
    helper function for making scheme with different number of pulses.

    :param n_pulses:
    :return: A decent IR acquisition scheme
    """
    tr = np.repeat(500, n_pulses)
    te = np.linspace(10, 20, n_pulses)
    ti = np.linspace(50, 400, n_pulses)
    return acquisition_scheme.InversionRecoveryAcquisitionScheme(tr, te, ti)


def make_repeated_scheme(n_pulses: int) -> acquisition_scheme.InversionRecoveryAcquisitionScheme:
    """
    helper function for making scheme with different number of pulses.

    :param n_pulses:
    :return: A not so decent IR acquisition scheme
    """
    tr = np.repeat(500, n_pulses)
    te = np.repeat(20, n_pulses)
    ti = np.repeat(400, n_pulses)
    return acquisition_scheme.InversionRecoveryAcquisitionScheme(tr, te, ti)


def compute_loss(model: tissue_model.TissueModel,
                 scheme: acquisition_scheme.AcquisitionScheme,
                 noise_var: float,
                 loss: optimize.LossFunction) -> float:
    """
    Function for computing the loss given the following parameters

    :param model: The tissuemodel for which you wish to know the loss
    :param scheme: The acquisition scheme for which you whish to know the loss
    :param noise_var:
    :param loss:
    :return:
    """
    model_scales = [value.scale for value in model.values()]
    model_include = [value.optimize for value in model.values()]
    jac = model.jacobian(scheme)
    return loss(jac, model_scales, model_include, noise_var)


# TODO: Parameterize test with different models and optimizers in the future
@pytest.mark.parametrize("n_pulses", [n for n in range(10, 1, -1)])
@pytest.mark.parametrize("scheme_factory", [make_scheme, make_repeated_scheme])
def test_relaxation(n_pulses: int, scheme_factory):
    """
    Testing if tissuemodel.optimize actually reduces the loss
    """
    # copying to prevent test interferences
    model = copy(RELAXATION_MODEL)
    scheme = scheme_factory(n_pulses)

    loss_non_optimal = compute_loss(model, scheme, NOISE, optimize.crlb_loss)

    result = model.optimize(scheme, NOISE)
    optimized_parameters = result['x'] * scheme.free_parameter_scales
    loss_optimal = compute_loss(model, scheme, NOISE, optimize.crlb_loss)
    assert loss_optimal < loss_non_optimal

from typing import Union, List

from scipy.optimize import LinearConstraint, NonlinearConstraint

from .utils.unit_registry import unit


GAMMA = 267.5987e3 * unit('1/mT/s')  # Proton gyromagnetic ratio [rad/mT/s].

B_UNIT = 's/mm²'
B_VAL_LB = 0.0 * unit('s/mm²')
B_VAL_UB = 3e3 * unit('s/mm²')
B_VAL_SCALE = 1e3 * unit('s/mm²')
B_MAX = 6000 * unit('s/mm²')  # TODO: what's the difference in usage with B_VAL_UB?

GRADIENT_UNIT = 'mT/mm'

PULSE_TIMING_UNIT = 's'
PULSE_TIMING_LB = 1e-3 * unit('s')
PULSE_TIMING_UB = 100e-3 * unit('s')
PULSE_TIMING_SCALE = 1e-2 * unit('s')
MAX_TE = 0.05 * unit('s')

SLEW_RATE_UNIT = 'mT/mm/s'

# Key for the starting signal
BASE_SIGNAL_KEY = "S0"

# MultiTissueModel constants
VOLUME_FRACTION_PREFIX = "vf_"
MODEL_PREFIX = "model_"

# Relaxation stuff
RELAXATION_PREFIX = "T2_relaxation_"
RELAXATION_BOUNDS = (.1e-3 * unit('s'), 100e-3 * unit('s'))

# old model constants
T2_KEY = 'T2'
T1_KEY = 'T1'

DIFFUSIVITY_KEY = 'Diffusivity'

ConstraintTypes = Union[
    List[Union[LinearConstraint, NonlinearConstraint]], Union[LinearConstraint, NonlinearConstraint]]

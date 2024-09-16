from __future__ import annotations

from typing import Union, List

import numpy as np
from scipy.optimize import LinearConstraint, NonlinearConstraint

from .utils.unit_registry import unit


# Base units that results in numeric values of order magnitude 1.
GAMMA = 42.57747892 * 2 * np.pi * 1e3 * unit('1/mT/s')  # Proton gyromagnetic ratio [1/mT/s].

B_UNIT = 's/mm²'
B_VAL_LB = 0.0  # [s/mm²]
B_VAL_UB = 3e3  # [s/mm²]
B_VAL_SCALE = 1e3  # [s/mm²]
B_MAX = 6000  # [s/mm²]  # TODO: what's the difference in usage with B_VAL_UB?

GRADIENT_UNIT = 'mT/mm'

PULSE_TIMING_UNIT = 's'
PULSE_TIMING_LB = 1e-3  # [s]
PULSE_TIMING_UB = 100e-3  # [s]
PULSE_TIMING_SCALE = 1e-2  # [s]
MAX_TE = 0.05  # [s]

SLEW_RATE_UNIT = 'mT/mm/s'

# Key for the starting signal
BASE_SIGNAL_KEY = "S0"

# MultiTissueModel constants
VOLUME_FRACTION_PREFIX = "vf_"
MODEL_PREFIX = "model_"

# Relaxation stuff
RELAXATION_PREFIX = "T2_relaxation_"
RELAXATION_BOUNDS = (.1e-3, 100e-3) #s

# old model constants
T2_KEY = 'T2'
T1_KEY = 'T1'
RELAXATION_BOUNDS = (.1e-3, 1) #s

DIFFUSIVITY_KEY = 'Diffusivity'

ConstraintTypes = Union[
    List[Union[LinearConstraint, NonlinearConstraint]], Union[LinearConstraint, NonlinearConstraint]]

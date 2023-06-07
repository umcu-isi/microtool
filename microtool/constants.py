from __future__ import annotations

from typing import Union, List

import numpy as np
from scipy.optimize import LinearConstraint, NonlinearConstraint

from . import Q_, ureg

# Key for the starting signal
BASE_SIGNAL_KEY = "S0"

# MultiTissueModel constants
VOLUME_FRACTION_PREFIX = "vf_"
MODEL_PREFIX = "model_"

# Relaxation stuff
RELAXATION_PREFIX = "T2_relaxation_"
RELAXATION_BOUNDS = (.1 * ureg.millisecond, 1e3 * ureg.millisecond)  # ms

# old model constants
T2_KEY = 'T2'
T1_KEY = 'T1'
DIFFUSIVITY_KEY = 'Diffusivity'

ConstraintTypes = Union[
    List[Union[LinearConstraint, NonlinearConstraint]], Union[LinearConstraint, NonlinearConstraint]]

GAMMA = Q_(42.57747892 * 2 * np.pi, 'MHz/T')

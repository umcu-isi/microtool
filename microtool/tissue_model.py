from dataclasses import dataclass
from typing import Optional

import numpy as np


class DiffusionModel:
    def __call__(self,
                 b_values: np.ndarray,
                 b_vectors: np.ndarray,
                 pulse_widths: np.ndarray,
                 pulse_intervals: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def jacobian(self,
                 b_values: np.ndarray,
                 b_vectors: np.ndarray,
                 pulse_widths: np.ndarray,
                 pulse_intervals: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


@dataclass
class TissueModel:
    t1: Optional[float] = None
    t2: Optional[float] = None
    diffusion_model: Optional[DiffusionModel] = None

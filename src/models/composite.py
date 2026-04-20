"""AWRY composite: P_AWRY = α · P_now + (1-α) · P_3m."""

from __future__ import annotations

import numpy as np


def composite_score(p_now: np.ndarray, p_3m: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return alpha * p_now + (1.0 - alpha) * p_3m

"""Shared lightweight model helpers."""

from __future__ import annotations

import numpy as np


class ConstantProbabilityModel:
    """Minimal classifier that always returns the same positive-class probability."""

    def __init__(self, p_positive: float) -> None:
        self.p_positive = float(np.clip(p_positive, 0.0, 1.0))
        self.classes_ = np.array([0, 1], dtype=int)

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X) -> np.ndarray:
        n = len(X)
        p1 = np.full(n, self.p_positive, dtype=float)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

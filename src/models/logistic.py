"""L2 logistic regression for recession probability."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    C: float = 1.0,
    class_weight: str | dict | None = "balanced",
    max_iter: int = 2000,
) -> LogisticRegression:
    model = LogisticRegression(
        C=C,
        solver="lbfgs",
        class_weight=class_weight,
        max_iter=max_iter,
    )
    model.fit(X, y)
    return model

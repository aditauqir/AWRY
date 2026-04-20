"""Random forest classifier with probability output."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def fit_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 300,
    max_depth: int | None = 8,
    min_samples_leaf: int = 5,
    class_weight: str | dict | None = "balanced",
    random_state: int = 42,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model

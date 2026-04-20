"""Backward-compatible wrapper around the capped random-forest pipeline."""

from __future__ import annotations

from models.rf import fit_rf


def fit_random_forest(
    X,
    y,
    n_estimators: int = 300,
    max_depth: int | None = 8,
    min_samples_leaf: int = 5,
    class_weight: str | dict | None = "balanced",
    random_state: int = 42,
):
    return fit_rf(X, y)

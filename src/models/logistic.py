"""Backward-compatible wrapper around the new regularized logistic pipeline."""

from __future__ import annotations

from models.logit import fit_logit


def fit_logistic(
    X,
    y,
    C: float = 1.0,
    class_weight: str | dict | None = "balanced",
    max_iter: int = 2000,
    cv=None,
):
    return fit_logit(X, y, cv=cv)

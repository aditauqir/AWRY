"""Stacked ensemble utilities for AWRY."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from models.common import ConstantProbabilityModel

EPS = 1e-6


def to_logit(p: np.ndarray) -> np.ndarray:
    """Convert probabilities to log-odds with safe clipping."""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def from_logit(z: np.ndarray) -> np.ndarray:
    """Convert log-odds back to probabilities."""
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def fit_meta_learner(Z: np.ndarray, y) -> LogisticRegression:
    """Fit the L2-regularized logistic meta learner."""
    y_arr = np.asarray(y).astype(int).reshape(-1)
    if len(np.unique(y_arr)) < 2:
        return ConstantProbabilityModel(float(y_arr.mean()))
    # Base model probabilities are highly correlated. An L2 penalty keeps
    # the stacker stable instead of swinging coefficients wildly fold to fold.
    meta = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="liblinear",
        max_iter=2000,
        random_state=42,
    )
    meta.fit(np.asarray(Z, dtype=float), y_arr)
    return meta


def stack_matrix(predictions: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Build the meta-learner design matrix from base probability vectors."""
    names = list(predictions.keys())
    cols = [to_logit(predictions[name]) for name in names]
    return np.column_stack(cols), names


def meta_predict(meta: LogisticRegression, predictions: dict[str, np.ndarray]) -> np.ndarray:
    """Predict calibrated stacker probabilities from base model probabilities."""
    Z, _ = stack_matrix(predictions)
    return meta.predict_proba(Z)[:, 1]


@dataclass
class StackedEnsemble:
    """Trained base models plus the fitted meta-learner."""

    base_models: dict[str, object]
    meta: LogisticRegression

    def predict_proba(self, X) -> np.ndarray:
        """Return stacker probabilities for the positive class."""
        preds = {name: model.predict_proba(X)[:, 1] for name, model in self.base_models.items()}
        return meta_predict(self.meta, preds)

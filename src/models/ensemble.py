"""Combine logistic and RF via log-odds weighting; weights from validation Brier."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def brier_optimal_weights(
    y_val: np.ndarray,
    p_logit: np.ndarray,
    p_rf: np.ndarray,
) -> tuple[float, float]:
    """Find w1, w2 (sum to 1) minimizing Brier on validation."""
    grid = np.linspace(0.0, 1.0, 101)
    best = (0.5, 0.5)
    best_score = float("inf")
    for w1 in grid:
        w2 = 1.0 - w1
        logit_combo = w1 * _logit(p_logit) + w2 * _logit(p_rf)
        p = _sigmoid(logit_combo)
        score = brier_score_loss(y_val, p)
        if score < best_score:
            best_score = score
            best = (w1, w2)
    return best


def ensemble_predict(w1: float, w2: float, p_logit: np.ndarray, p_rf: np.ndarray) -> np.ndarray:
    logit_combo = w1 * _logit(p_logit) + w2 * _logit(p_rf)
    return _sigmoid(logit_combo)

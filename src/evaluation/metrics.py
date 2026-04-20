"""Brier, AUROC, F1."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score


def evaluate_binary(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    out: dict[str, float] = {
        "brier": float(brier_score_loss(y_true, y_prob)),
    }
    if len(np.unique(y_true)) > 1:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
    else:
        out["auroc"] = float("nan")
    y_hat = (y_prob >= 0.5).astype(int)
    out["f1"] = float(f1_score(y_true, y_hat, zero_division=0))
    return out

"""Optional XGBoost builder for the stacked AWRY ensemble."""

from __future__ import annotations

import numpy as np

from models.common import ConstantProbabilityModel

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - depends on optional package
    XGBClassifier = None  # type: ignore


def build_xgb():
    """Build the constrained XGBoost classifier."""
    if XGBClassifier is None:
        raise RuntimeError("xgboost is not installed. Add it to the environment before running the stacker.")
    # These constraints intentionally bias toward smoother fits because
    # recession data is small-sample and heavily imbalanced.
    return XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )


def has_xgboost() -> bool:
    return XGBClassifier is not None


def fit_xgb(X_fit, y_fit, X_val=None, y_val=None):
    """Fit XGBoost with a constant-model fallback for single-class windows."""
    y_arr = np.asarray(y_fit).astype(int)
    if len(np.unique(y_arr)) < 2:
        return ConstantProbabilityModel(float(y_arr.mean()))
    model = build_xgb()
    if X_val is not None and y_val is not None and len(X_val) > 0:
        model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_fit, y_fit, verbose=False)
    return model

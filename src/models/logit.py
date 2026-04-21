"""Regularized logistic regression builders for AWRY."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.common import ConstantProbabilityModel


def build_logit() -> Pipeline:
    """Build the regularized logistic pipeline used in every horizon."""
    # We standardize inside the pipeline so every fold learns its scaler
    # only from the training window.
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


def fit_logit(X, y, cv=None) -> Pipeline:
    """Fit the regularized logistic pipeline."""
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return ConstantProbabilityModel(float(y.mean()))
    model = build_logit()
    try:
        if cv is None:
            model.fit(X, y)
            return model
        search = GridSearchCV(
            estimator=model,
            param_grid={"model__C": [0.01, 0.1, 1.0, 10.0]},
            scoring="neg_brier_score",
            cv=cv,
            refit=True,
            n_jobs=1,
            error_score="raise",
        )
        search.fit(X, y)
        return search.best_estimator_
    except ValueError:
        # Small early folds can still break the inner CV if one split lands
        # on a single-class window. Fall back to the tuned model family without CV.
        fallback = clone(model)
        fallback.fit(X, y)
        return fallback

"""End-to-end: build table, train horizon models, ensemble, composite."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd

from features.dataset_builder import build_model_table, feature_matrix_columns
from features.equity_config import DEFAULT_EQUITY_SERIES
from models.composite import composite_score
from models.ensemble import brier_optimal_weights, ensemble_predict
from models.logistic import fit_logistic
from models.random_forest import fit_random_forest


@dataclass
class HorizonModels:
    horizon: int
    logit: object
    rf: object
    w1: float
    w2: float


@dataclass
class AwryPipeline:
    df: pd.DataFrame
    x_cols: list[str]
    now: HorizonModels
    forecast3: HorizonModels
    alpha: float

    def predict_row(self, X: np.ndarray) -> tuple[float, float, float]:
        """Returns (p_now, p_3m, p_awry)."""
        X = X.reshape(1, -1)
        p0_l = self.now.logit.predict_proba(X)[0, 1]
        p0_r = self.now.rf.predict_proba(X)[0, 1]
        p_now = ensemble_predict(self.now.w1, self.now.w2, np.array([p0_l]), np.array([p0_r]))[0]
        p3_l = self.forecast3.logit.predict_proba(X)[0, 1]
        p3_r = self.forecast3.rf.predict_proba(X)[0, 1]
        p_3m = ensemble_predict(self.forecast3.w1, self.forecast3.w2, np.array([p3_l]), np.array([p3_r]))[0]
        p_awry = composite_score(np.array([p_now]), np.array([p_3m]), self.alpha)[0]
        return float(p_now), float(p_3m), float(p_awry)


def train_horizon(
    train: pd.DataFrame,
    val: pd.DataFrame,
    x_cols: list[str],
    target_col: str,
) -> HorizonModels:
    X_tr = train[x_cols].values
    y_tr = train[target_col].values.astype(int)
    X_va = val[x_cols].values
    y_va = val[target_col].values.astype(int)

    logit = fit_logistic(X_tr, y_tr)
    rf = fit_random_forest(X_tr, y_tr)

    p_l_va = logit.predict_proba(X_va)[:, 1]
    p_r_va = rf.predict_proba(X_va)[:, 1]
    w1, w2 = brier_optimal_weights(y_va, p_l_va, p_r_va)

    hz = int(target_col.replace("target_h", ""))
    return HorizonModels(horizon=hz, logit=logit, rf=rf, w1=w1, w2=w2)


def fit_awry_pipeline(
    val_fraction: float = 0.15,
    alpha: float = 0.5,
    cached: bool = True,
    equity_series: str | None = None,
) -> AwryPipeline:
    eq = equity_series or DEFAULT_EQUITY_SERIES
    df = build_model_table(cached=cached, equity_series=eq)
    x_cols = feature_matrix_columns(df)
    n = len(df)
    split = int(n * (1.0 - val_fraction))
    train = df.iloc[:split]
    val = df.iloc[split:]

    now = train_horizon(train, val, x_cols, "target_h0")
    forecast3 = train_horizon(train, val, x_cols, "target_h3")

    return AwryPipeline(df=df, x_cols=x_cols, now=now, forecast3=forecast3, alpha=alpha)


def predict_history(pipe: AwryPipeline) -> pd.DataFrame:
    """In-sample / full-series probabilities (for charting)."""
    X = pipe.df[pipe.x_cols].values
    pl_now = ensemble_predict(
        pipe.now.w1,
        pipe.now.w2,
        pipe.now.logit.predict_proba(X)[:, 1],
        pipe.now.rf.predict_proba(X)[:, 1],
    )
    pl_3 = ensemble_predict(
        pipe.forecast3.w1,
        pipe.forecast3.w2,
        pipe.forecast3.logit.predict_proba(X)[:, 1],
        pipe.forecast3.rf.predict_proba(X)[:, 1],
    )
    p_awry = composite_score(pl_now, pl_3, pipe.alpha)
    out = pipe.df[["USREC"]].copy()
    out["P_now"] = pl_now
    out["P_3m"] = pl_3
    out["P_AWRY"] = p_awry
    if "target_h0" in pipe.df.columns:
        out["target_h0"] = pipe.df["target_h0"].values
    if "target_h3" in pipe.df.columns:
        out["target_h3"] = pipe.df["target_h3"].values
    return out

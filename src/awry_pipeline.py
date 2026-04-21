"""End-to-end training and OOF evaluation pipeline for AWRY."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    # The dashboard can be launched as a module or as a script, so we
    # add src/ explicitly to keep local imports reliable in both paths.
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd

from evaluation.walk_forward import run_full_walk_forward
from features.dataset_builder import feature_matrix_columns
from features.equity_config import DEFAULT_EQUITY_SERIES


@dataclass
class HorizonModels:
    """A fitted horizon-specific stacker plus metadata for exports."""

    horizon: int
    ensemble: object
    base_names: list[str]
    fixed_weights: tuple[float, float] | None
    metrics: dict[str, float]
    meta_coefficients: dict[str, float]

    @property
    def logit(self):
        return self.ensemble.base_models.get("logit")

    @property
    def rf(self):
        return self.ensemble.base_models.get("rf")

    @property
    def xgb(self):
        return self.ensemble.base_models.get("xgb")

    @property
    def w1(self) -> float:
        return float(self.fixed_weights[0]) if self.fixed_weights else float("nan")

    @property
    def w2(self) -> float:
        return float(self.fixed_weights[1]) if self.fixed_weights else float("nan")

    def predict_proba(self, X) -> np.ndarray:
        return np.asarray(self.ensemble.predict_proba(X), dtype=float)


@dataclass
class AwryPipeline:
    """Final fitted horizon models plus OOF and in-sample histories."""

    df: pd.DataFrame
    x_cols: list[str]
    now: HorizonModels
    forecast3: HorizonModels
    alpha: float
    oof_history: pd.DataFrame
    full_history: pd.DataFrame
    thresholds: dict[str, float]
    summary: dict[str, Any]

    def predict_row(self, X: np.ndarray) -> tuple[float, float, float]:
        """Return (p_now, p_3m, p_awry) for a single row."""
        X_df = pd.DataFrame([X], columns=self.x_cols)
        p_now = self.now.predict_proba(X_df)[0]
        p_3m = self.forecast3.predict_proba(X_df)[0]
        p_awry = self.alpha * p_now + (1.0 - self.alpha) * p_3m
        return float(p_now), float(p_3m), float(p_awry)


def _build_full_history(df: pd.DataFrame, x_cols: list[str], now: HorizonModels, forecast3: HorizonModels, alpha: float) -> pd.DataFrame:
    X = df[x_cols]
    out = df[["USREC"]].copy()
    out["P_now"] = now.predict_proba(X)
    out["P_3m"] = forecast3.predict_proba(X)
    out["P_AWRY"] = alpha * out["P_now"] + (1.0 - alpha) * out["P_3m"]
    if "target_h0" in df.columns:
        out["target_h0"] = df["target_h0"].values
    if "target_h3" in df.columns:
        out["target_h3"] = df["target_h3"].values
    return out


def fit_awry_pipeline(
    val_fraction: float = 0.15,
    alpha: float | None = None,
    cached: bool = True,
    equity_series: str | None = None,
    feature_set: str = "baseline",
    save_artifacts: bool = True,
) -> AwryPipeline:
    """Fit the horizon stackers and compute OOF artifacts for the dashboard."""
    del val_fraction  # The new pipeline uses purged walk-forward CV instead.
    eq = equity_series or DEFAULT_EQUITY_SERIES
    # This is the main handoff from the app/reporting layer into the
    # research pipeline: build features, run walk-forward CV, tune alpha, and
    # return both fitted models and the saved artifact tables.
    result = run_full_walk_forward(
        cached=cached,
        equity_series=eq,
        feature_set=feature_set,
        save_artifacts=save_artifacts,
    )
    chosen_alpha = float(result["summary"]["alpha"] if alpha is None else alpha)
    now_eval = result["now"]
    forecast_eval = result["forecast"]

    now = HorizonModels(
        horizon=0,
        ensemble=now_eval.final_ensemble,
        base_names=now_eval.base_names,
        fixed_weights=now_eval.fixed_weights,
        metrics=now_eval.metrics,
        meta_coefficients=now_eval.meta_coefficients,
    )
    forecast3 = HorizonModels(
        horizon=3,
        ensemble=forecast_eval.final_ensemble,
        base_names=forecast_eval.base_names,
        fixed_weights=forecast_eval.fixed_weights,
        metrics=forecast_eval.metrics,
        meta_coefficients=forecast_eval.meta_coefficients,
    )

    df = now_eval.df.copy()
    x_cols = feature_matrix_columns(df, feature_set=feature_set)
    x_cols = [col for col in x_cols if col in df.columns and col in now_eval.x_cols]
    reference_history = result.get("composite_reference")
    if isinstance(reference_history, pd.DataFrame) and not reference_history.empty:
        # Reference history is useful for dashboard timelines, but the
        # OOF history remains the honest out-of-sample evidence for the paper.
        full_history = reference_history.copy()
    else:
        full_history = _build_full_history(df, x_cols, now, forecast3, chosen_alpha)

    oof_history = result["composite_oof"].copy()
    if chosen_alpha != float(result["summary"]["alpha"]):
        oof_history["P_AWRY"] = chosen_alpha * oof_history["P_now"] + (1.0 - chosen_alpha) * oof_history["P_3m"]

    summary = dict(result["summary"])
    summary["alpha"] = chosen_alpha

    return AwryPipeline(
        df=df,
        x_cols=x_cols,
        now=now,
        forecast3=forecast3,
        alpha=chosen_alpha,
        oof_history=oof_history,
        full_history=full_history,
        thresholds=summary["thresholds"],
        summary=summary,
    )


def predict_history(pipe: AwryPipeline, *, use_oof: bool = True) -> pd.DataFrame:
    """Return the OOF history by default; full-history predictions remain available for reference."""
    return pipe.oof_history.copy() if use_oof else pipe.full_history.copy()


def train_horizon(*args, **kwargs):  # pragma: no cover - compatibility shim
    raise RuntimeError("train_horizon has been replaced by the purged walk-forward stacker path.")

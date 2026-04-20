"""Purged walk-forward evaluation and OOF artifact generation for AWRY."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit

from config import FIGURES_DIR, MODELS_DIR, OOF_PRED_DIR, PROJECT_ROOT
from dashboard.components.backtest_chart import (
    AWRY_BACKTEST_SIGNAL_THRESHOLD,
    SCENARIOS,
    lead_months_awry,
)
from evaluation.metrics import evaluate_binary
from features.dataset_builder import build_model_table, feature_matrix_columns, redundant_lag_columns
from models.ensemble import brier_optimal_weights, ensemble_predict
from models.logit import fit_logit
from models.rf import fit_rf
from models.stacker import StackedEnsemble, fit_meta_learner, meta_predict
from models.xgb import fit_xgb, has_xgboost

HORIZON_LABELS = {
    "target_h0": "nowcast",
    "target_h3": "forecast3",
}


@dataclass
class HorizonEvaluation:
    """Artifacts for one horizon's base models and stacker."""

    horizon: str
    feature_set: str
    df: pd.DataFrame
    x_cols: list[str]
    base_names: list[str]
    oof: pd.DataFrame
    final_ensemble: StackedEnsemble
    fixed_weights: tuple[float, float] | None
    metrics: dict[str, float]
    meta_coefficients: dict[str, float]
    fold_metrics: pd.DataFrame
    dropped_lags: list[str]


def make_cv(n_splits: int = 5, gap: int = 3) -> TimeSeriesSplit:
    """Build the purged walk-forward splitter used everywhere downstream."""
    # COMMENT: target_h3 = USREC.shift(-3), so a 3-month gap is required to stop
    # training labels from containing information about the first 3 test months.
    return TimeSeriesSplit(n_splits=n_splits, gap=gap)


def _inner_cv(train_size: int, gap: int = 3) -> TimeSeriesSplit:
    n_splits = 3 if train_size >= 96 else 2
    return TimeSeriesSplit(n_splits=n_splits, gap=min(gap, max(0, train_size // 8)))


def _split_for_xgb_validation(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split = max(1, int(len(X) * 0.85))
    if split >= len(X):
        split = len(X) - 1
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def _fit_base_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    use_xgb: bool = True,
) -> dict[str, Any]:
    inner = _inner_cv(len(X_train))
    models: dict[str, Any] = {
        "logit": fit_logit(X_train, y_train, cv=inner),
        "rf": fit_rf(X_train, y_train),
    }
    if use_xgb and has_xgboost():
        X_fit, X_val, y_fit, y_val = _split_for_xgb_validation(X_train, y_train)
        xgb = fit_xgb(X_fit, y_fit, X_val, y_val)
        models["xgb"] = xgb
    return models


def _oof_frame(index: pd.Index, y: pd.Series, horizon: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(index),
            "y_true": np.asarray(y).astype(int),
            "fold_idx": -1,
            "horizon": horizon,
        }
    ).set_index("date")


def _save_parquet(df: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def _save_json(payload: dict[str, Any], path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _fixed_ensemble_predictions(df: pd.DataFrame, target_col: str) -> tuple[np.ndarray, tuple[float, float]]:
    w1, w2 = brier_optimal_weights(df["y_true"].values, df["logit"].values, df["rf"].values)
    p = ensemble_predict(w1, w2, df["logit"].values, df["rf"].values)
    return p, (float(w1), float(w2))


def run_horizon_walk_forward(
    *,
    target_col: str,
    cached: bool = True,
    equity_series: str | None = None,
    feature_set: str = "full",
    n_splits: int = 5,
    gap: int = 3,
    save_artifacts: bool = True,
) -> HorizonEvaluation:
    """Generate OOF predictions and a final stacker for one target horizon."""
    df = build_model_table(cached=cached, equity_series=equity_series, feature_set=feature_set)
    x_cols = feature_matrix_columns(df, feature_set=feature_set)
    dropped = redundant_lag_columns(df[x_cols])
    x_cols = [col for col in x_cols if col not in dropped]

    X = df[x_cols].copy()
    y = df[target_col].astype(int).copy()
    cv = make_cv(n_splits=n_splits, gap=gap)

    oof = _oof_frame(df.index, y, target_col)
    fold_rows: list[dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X), start=1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        base_models = _fit_base_models(X_train, y_train, use_xgb=True)
        fold_preds: dict[str, np.ndarray] = {}
        for name, model in base_models.items():
            p = model.predict_proba(X_test)[:, 1]
            fold_preds[name] = p
            oof.loc[X_test.index, name] = p

        oof.loc[X_test.index, "fold_idx"] = fold_idx

        if {"logit", "rf"} <= set(fold_preds):
            fixed_p = ensemble_predict(0.5, 0.5, fold_preds["logit"], fold_preds["rf"])
            fixed_metrics = evaluate_binary(y_test.values, fixed_p)
        else:
            fixed_metrics = {"brier": float("nan"), "auroc": float("nan"), "f1": float("nan")}

        fold_rows.append(
            {
                "fold_idx": fold_idx,
                "train_start": str(X_train.index[0].date()),
                "train_end": str(X_train.index[-1].date()),
                "test_start": str(X_test.index[0].date()),
                "test_end": str(X_test.index[-1].date()),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "brier_fixed": float(fixed_metrics["brier"]),
                "auroc_fixed": float(fixed_metrics["auroc"]),
                "f1_fixed": float(fixed_metrics["f1"]),
            }
        )

    predicted_cols = [col for col in ["logit", "rf", "xgb"] if col in oof.columns]
    oof = oof.loc[oof["fold_idx"] >= 0].copy()

    fixed_weights: tuple[float, float] | None = None
    if {"logit", "rf"} <= set(predicted_cols):
        oof["fixed"] = np.nan
        oof["fixed"], fixed_weights = _fixed_ensemble_predictions(oof, target_col)

    Z = oof[predicted_cols].copy()
    meta = fit_meta_learner(np.log(np.clip(Z.values, 1e-6, 1.0 - 1e-6) / (1.0 - np.clip(Z.values, 1e-6, 1.0 - 1e-6))), y.loc[oof.index])
    oof["stacked"] = meta_predict(meta, {name: oof[name].values for name in predicted_cols})

    final_models = _fit_base_models(X, y, use_xgb=True)
    final_ensemble = StackedEnsemble(base_models=final_models, meta=meta)

    metrics = evaluate_binary(oof["y_true"].values, oof["stacked"].values)
    if hasattr(meta, "intercept_") and hasattr(meta, "coef_"):
        coefs = {"intercept": float(meta.intercept_[0])}
        for idx, name in enumerate(predicted_cols):
            coefs[name] = float(meta.coef_[0][idx])
    else:
        coefs = {"intercept": float(getattr(meta, "p_positive", 0.0))}
        for name in predicted_cols:
            coefs[name] = 0.0

    fold_metrics = pd.DataFrame(fold_rows)

    label = HORIZON_LABELS.get(target_col, target_col)
    if save_artifacts:
        base_path = OOF_PRED_DIR
        for name in predicted_cols + ["stacked"] + (["fixed"] if "fixed" in oof.columns else []):
            cols = ["fold_idx", "y_true", name]
            artifact = oof[cols].reset_index().rename(columns={"index": "date", name: "y_pred"})
            _save_parquet(artifact, base_path / f"{label}_{name}_{feature_set}.parquet")

        _save_json(
            {
                "target": target_col,
                "feature_set": feature_set,
                "metrics": metrics,
                "meta_coefficients": coefs,
                "fixed_weights": fixed_weights,
                "dropped_lags": dropped,
                "x_cols": x_cols,
            },
            MODELS_DIR / f"{label}_{feature_set}_metrics.json",
        )

    return HorizonEvaluation(
        horizon=target_col,
        feature_set=feature_set,
        df=df,
        x_cols=x_cols,
        base_names=predicted_cols,
        oof=oof,
        final_ensemble=final_ensemble,
        fixed_weights=fixed_weights,
        metrics=metrics,
        meta_coefficients=coefs,
        fold_metrics=fold_metrics,
        dropped_lags=dropped,
    )


def tune_alpha(
    now_eval: HorizonEvaluation,
    forecast_eval: HorizonEvaluation,
    *,
    feature_set: str = "full",
    save_artifacts: bool = True,
) -> tuple[float, pd.DataFrame]:
    """Search alpha on OOF horizon predictions and save the curve."""
    common = now_eval.oof.index.intersection(forecast_eval.oof.index)
    frame = pd.DataFrame(
        {
            "y_true": now_eval.oof.loc[common, "y_true"].astype(int),
            "P_now": now_eval.oof.loc[common, "stacked"].astype(float),
            "P_3m": forecast_eval.oof.loc[common, "stacked"].astype(float),
        },
        index=common,
    ).sort_index()

    rows: list[dict[str, float]] = []
    best_alpha = 0.5
    best_brier = float("inf")
    for alpha in np.linspace(0.0, 1.0, 21):
        p = alpha * frame["P_now"].values + (1.0 - alpha) * frame["P_3m"].values
        brier = brier_score_loss(frame["y_true"].values, p)
        ap = average_precision_score(frame["y_true"].values, p)
        rows.append({"alpha": float(alpha), "brier": float(brier), "average_precision": float(ap)})
        if brier < best_brier:
            best_brier = float(brier)
            best_alpha = float(alpha)

    alpha_df = pd.DataFrame(rows)
    if save_artifacts:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(alpha_df["alpha"], alpha_df["brier"], marker="o", color="#2563eb")
        ax.axvline(best_alpha, color="#dc2626", linestyle="--", label=f"best alpha={best_alpha:.2f}")
        ax.set_title("OOF Brier by alpha")
        ax.set_xlabel("alpha")
        ax.set_ylabel("Brier score")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "alpha_curve.png", dpi=180)
        plt.close(fig)

    return best_alpha, alpha_df


def build_composite_oof(
    now_eval: HorizonEvaluation,
    forecast_eval: HorizonEvaluation,
    *,
    alpha: float,
    feature_set: str = "full",
    save_artifacts: bool = True,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Blend OOF horizon probabilities into the headline AWRY composite."""
    common = now_eval.oof.index.intersection(forecast_eval.oof.index)
    out = pd.DataFrame(index=common)
    out["fold_idx"] = now_eval.oof.loc[common, "fold_idx"].astype(int)
    out["USREC"] = now_eval.oof.loc[common, "y_true"].astype(int)
    out["target_h0"] = now_eval.oof.loc[common, "y_true"].astype(int)
    out["target_h3"] = forecast_eval.oof.loc[common, "y_true"].astype(int)
    out["P_now"] = now_eval.oof.loc[common, "stacked"].astype(float)
    out["P_3m"] = forecast_eval.oof.loc[common, "stacked"].astype(float)
    out["P_AWRY"] = alpha * out["P_now"] + (1.0 - alpha) * out["P_3m"]
    metrics = evaluate_binary(out["USREC"].values, out["P_AWRY"].values)

    if save_artifacts:
        _save_parquet(out.reset_index().rename(columns={"index": "date"}), OOF_PRED_DIR / "composite_oof.parquet")
        _save_json({"alpha": alpha, "metrics": metrics}, MODELS_DIR / f"composite_{feature_set}_metrics.json")

    return out, metrics


def build_composite_reference_history(
    now_eval: HorizonEvaluation,
    forecast_eval: HorizonEvaluation,
    *,
    alpha: float,
) -> pd.DataFrame:
    """Build fitted full-history probabilities for scenario reference views."""
    common_cols = [col for col in now_eval.x_cols if col in forecast_eval.df.columns and col in now_eval.df.columns]
    X_now = now_eval.df[common_cols].copy()
    X_fc = forecast_eval.df[common_cols].copy()

    common_index = now_eval.df.index.intersection(forecast_eval.df.index)
    X_now = X_now.loc[common_index]
    X_fc = X_fc.loc[common_index]

    out = pd.DataFrame(index=common_index)
    out["USREC"] = now_eval.df.loc[common_index, "USREC"].astype(int)
    out["target_h0"] = now_eval.df.loc[common_index, "target_h0"].astype(int)
    out["target_h3"] = forecast_eval.df.loc[common_index, "target_h3"].astype(int)
    out["P_now"] = now_eval.final_ensemble.predict_proba(X_now)
    out["P_3m"] = forecast_eval.final_ensemble.predict_proba(X_fc)
    out["P_AWRY"] = alpha * out["P_now"] + (1.0 - alpha) * out["P_3m"]
    return out


def select_thresholds(
    composite_oof: pd.DataFrame,
    *,
    save_artifacts: bool = True,
) -> dict[str, float]:
    """Choose the operating threshold from the OOF precision-recall curve."""
    y_true = composite_oof["USREC"].values.astype(int)
    y_score = composite_oof["P_AWRY"].values.astype(float)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    thresholds = np.append(thresholds, 1.0)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    best_idx = int(np.nanargmax(f1))
    payload = {
        "threshold": float(thresholds[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "f1": float(f1[best_idx]),
        "class_imbalance": float(y_true.mean()),
    }

    if save_artifacts:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(recall, precision, color="#2563eb")
        ax.scatter(recall[best_idx], precision[best_idx], color="#dc2626", label=f"best tau={payload['threshold']:.2f}")
        ax.set_title("OOF Precision-Recall Curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "threshold_pr_curve.png", dpi=180)
        plt.close(fig)
        _save_json(payload, MODELS_DIR / "thresholds.json")

    return payload


def build_in_sample_reference(
    df: pd.DataFrame,
    ensemble: StackedEnsemble,
    x_cols: list[str],
    target_col: str,
) -> dict[str, float]:
    """Compute full-sample reference metrics for disclosure only."""
    probs = ensemble.predict_proba(df[x_cols])
    metrics = evaluate_binary(df[target_col].values, probs)
    _save_json(metrics, MODELS_DIR / "in_sample_metrics.json")
    return metrics


def build_backtest_summary(hist: pd.DataFrame) -> dict[str, int | None]:
    """Compute the AWRY lead time for each named recession scenario."""
    return {
        name: lead_months_awry(hist, name, threshold=AWRY_BACKTEST_SIGNAL_THRESHOLD)
        for name in SCENARIOS
    }


def run_full_walk_forward(
    *,
    cached: bool = True,
    equity_series: str | None = None,
    feature_set: str = "full",
    save_artifacts: bool = True,
) -> dict[str, Any]:
    """Run both horizon evaluations and build the composite OOF artifact set."""
    now_eval = run_horizon_walk_forward(
        target_col="target_h0",
        cached=cached,
        equity_series=equity_series,
        feature_set=feature_set,
        save_artifacts=save_artifacts,
    )
    forecast_eval = run_horizon_walk_forward(
        target_col="target_h3",
        cached=cached,
        equity_series=equity_series,
        feature_set=feature_set,
        save_artifacts=save_artifacts,
    )
    alpha, alpha_df = tune_alpha(now_eval, forecast_eval, feature_set=feature_set, save_artifacts=save_artifacts)
    composite_oof, composite_metrics = build_composite_oof(
        now_eval,
        forecast_eval,
        alpha=alpha,
        feature_set=feature_set,
        save_artifacts=save_artifacts,
    )
    composite_reference = build_composite_reference_history(
        now_eval,
        forecast_eval,
        alpha=alpha,
    )
    thresholds = select_thresholds(composite_oof, save_artifacts=save_artifacts)
    in_sample = build_in_sample_reference(now_eval.df, now_eval.final_ensemble, now_eval.x_cols, "target_h0")
    summary = {
        "feature_set": feature_set,
        "project_root": str(PROJECT_ROOT),
        "now_metrics": now_eval.metrics,
        "forecast_metrics": forecast_eval.metrics,
        "composite_metrics": composite_metrics,
        "alpha": alpha,
        "thresholds": thresholds,
        "backtest_leads": build_backtest_summary(composite_reference),
        "in_sample_metrics": in_sample,
    }
    if save_artifacts:
        _save_json(summary, MODELS_DIR / f"walk_forward_summary_{feature_set}.json")
        _save_parquet(alpha_df, OOF_PRED_DIR / f"alpha_curve_{feature_set}.parquet")
        _save_parquet(
            composite_reference.reset_index().rename(columns={"index": "date"}),
            OOF_PRED_DIR / f"composite_reference_{feature_set}.parquet",
        )
    return {
        "now": now_eval,
        "forecast": forecast_eval,
        "composite_oof": composite_oof,
        "composite_reference": composite_reference,
        "alpha_df": alpha_df,
        "summary": summary,
    }

"""
Diagnostic checks for post-fix walk-forward CV behavior.

Do not re-tune anything based on these outputs. These are reporting tools only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

ARTIFACTS = Path("artifacts")
FIGURES = ARTIFACTS / "figures"
OOF_PREDS = ARTIFACTS / "oof_preds"
MODELS = ARTIFACTS / "models"
FITTED_THRESHOLD = 0.2325
THRESHOLDS = [0.10, 0.15, 0.20, FITTED_THRESHOLD, 0.30]
RECESSION_STARTS = {
    "2001_dotcom": "2001-03-31",
    "2008_gfc": "2007-12-31",
    "2020_covid": "2020-02-29",
}


def _prepare_diagnostic_dirs() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)


def _as_month_end_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
        out = out.set_index("date")
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def _load_oof_predictions_and_labels() -> tuple[pd.Series, pd.Series, pd.DataFrame, str]:
    oof_path = OOF_PREDS / "composite_oof.parquet"
    if not oof_path.exists():
        raise FileNotFoundError(
            f"Missing required OOF predictions at {oof_path}. "
            "Run the model artifact generation first; diagnostics will not fabricate inputs."
        )

    oof = _as_month_end_index(pd.read_parquet(oof_path))
    if "P_AWRY" in oof.columns:
        composite = oof["P_AWRY"].astype(float).rename("P_AWRY")
    elif "p_awry" in oof.columns:
        composite = oof["p_awry"].astype(float).rename("P_AWRY")
    else:
        numeric_cols = oof.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError(f"No numeric probability columns found in {oof_path}.")
        composite = oof[numeric_cols[0]].astype(float).rename("P_AWRY")

    y_path = OOF_PREDS / "y_true.parquet"
    if y_path.exists():
        y_df = _as_month_end_index(pd.read_parquet(y_path))
        label_col = "target" if "target" in y_df.columns else y_df.columns[0]
        y = y_df[label_col].astype(int).rename("target")
        label_source = str(y_path)
    elif "USREC" in oof.columns:
        y = oof["USREC"].astype(int).rename("target")
        label_source = f"{oof_path}::USREC"
    elif "target_h0" in oof.columns:
        y = oof["target_h0"].astype(int).rename("target")
        label_source = f"{oof_path}::target_h0"
    else:
        raise FileNotFoundError(
            f"Missing {y_path} and no USREC/target_h0 labels were found in {oof_path}. "
            "Diagnostics require real labels and will not fabricate them."
        )

    common = composite.index.intersection(y.index)
    if common.empty:
        raise ValueError("OOF probabilities and true labels have no overlapping dates.")
    return composite.loc[common], y.loc[common], oof.loc[common], label_source


def report_fold_positive_counts(y: pd.Series, n_splits: int = 5, gap: int = 3) -> pd.DataFrame:
    """
    For each walk-forward fold, report:
      - train positive count, train positive rate
      - test positive count, test positive rate

    A test fold with 0 positives makes AUROC/F1 undefined for that fold.
    If recession months cluster in a single fold, the apparent generalization
    of the other folds can be misleadingly high.
    """
    cv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    rows = []
    for i, (train_idx, test_idx) in enumerate(cv.split(y)):
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        rows.append(
            {
                "fold": i,
                "train_start": y.index[train_idx[0]],
                "train_end": y.index[train_idx[-1]],
                "test_start": y.index[test_idx[0]],
                "test_end": y.index[test_idx[-1]],
                "train_n": len(train_idx),
                "train_pos": int(y_train.sum()),
                "train_pos_rate": float(y_train.mean()),
                "test_n": len(test_idx),
                "test_pos": int(y_test.sum()),
                "test_pos_rate": float(y_test.mean()),
            }
        )
    df = pd.DataFrame(rows)
    print("\n=== Fold positive counts ===")
    print(df.to_string(index=False))
    df.to_csv(FIGURES / "diagnostic_fold_counts.csv", index=False)
    return df


def report_pre_recession_probability_peaks(
    oof_composite: pd.Series,
    recession_starts: dict[str, str],
    lookback_months: int = 12,
) -> pd.DataFrame:
    """
    For each recession, report the peak OOF composite probability in the
    N months before R0. This shows whether "no signal" means the model saw
    nothing or the model almost crossed the fitted threshold.
    """
    rows = []
    for name, r0 in recession_starts.items():
        r0_ts = pd.Timestamp(r0)
        window = oof_composite.loc[
            r0_ts - pd.DateOffset(months=lookback_months) : r0_ts - pd.DateOffset(months=1)
        ]
        if len(window) == 0:
            rows.append(
                {
                    "scenario": name,
                    "r0": r0,
                    "window_start": None,
                    "window_end": None,
                    "peak_pre_r0": None,
                    "peak_month": None,
                    "peak_to_threshold_gap": None,
                }
            )
            continue
        rows.append(
            {
                "scenario": name,
                "r0": r0,
                "window_start": window.index[0],
                "window_end": window.index[-1],
                "peak_pre_r0": float(window.max()),
                "peak_month": str(window.idxmax().date()),
                "peak_to_threshold_gap": float(window.max() - FITTED_THRESHOLD),
            }
        )
    df = pd.DataFrame(rows)
    print("\n=== Pre-recession probability peaks ===")
    print(df.to_string(index=False))
    df.to_csv(FIGURES / "diagnostic_peaks.csv", index=False)
    return df


def lead_time_threshold_sweep(
    oof_composite: pd.Series,
    recession_starts: dict[str, str],
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    For each scenario/threshold pair, compute lead time.

    This shows the sensitivity of lead-time claims to threshold choice instead
    of reporting only one threshold.
    """
    thresholds = thresholds or [0.10, 0.15, 0.20, FITTED_THRESHOLD, 0.30, 0.50]
    rows = []
    for name, r0 in recession_starts.items():
        r0_ts = pd.Timestamp(r0)
        # Only look at the 24 months before R0 for threshold crossings.
        # Crossings far before that can reflect prior stress episodes instead
        # of a useful signal for the named recession.
        window = oof_composite.loc[r0_ts - pd.DateOffset(months=24) : r0_ts]
        for tau in thresholds:
            crosses = window[window >= tau]
            if len(crosses) == 0:
                lead = None
            else:
                first_cross = crosses.index[0]
                lead = (r0_ts.year - first_cross.year) * 12 + (r0_ts.month - first_cross.month)
                lead = -lead
            rows.append(
                {
                    "scenario": name,
                    "threshold": tau,
                    "lead_months": lead,
                }
            )
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="scenario", columns="threshold", values="lead_months")
    print("\n=== Lead time by threshold ===")
    print(pivot.to_string())
    pivot.to_csv(FIGURES / "diagnostic_threshold_sweep.csv")
    return pivot


def inspect_stacker_coefficients(stacker_path: Path) -> pd.DataFrame:
    """
    Print the meta-learner's coefficients.

    This repo does not currently persist a stacker.joblib file, so the fallback
    reads the saved horizon metrics JSON files that contain the same meta
    coefficient payload emitted by the training pipeline.
    """
    rows: list[dict[str, float | str]] = []
    print("\n=== Stacker meta-learner coefficients ===")

    if stacker_path.exists():
        import joblib

        model = joblib.load(stacker_path)
        ml = model.meta_learner_ if hasattr(model, "meta_learner_") else model
        if hasattr(ml, "coef_"):
            print(f"Source: {stacker_path}")
            print(f"Coefficients: {ml.coef_}")
            print(f"Intercept: {ml.intercept_}")
            rows.append({"source": str(stacker_path), "term": "intercept", "value": float(ml.intercept_[0])})
            for idx, value in enumerate(ml.coef_.reshape(-1)):
                rows.append({"source": str(stacker_path), "term": f"coef_{idx}", "value": float(value)})
        else:
            print(f"Model type: {type(ml).__name__}")
            print(f"Model: {ml}")
            rows.append({"source": str(stacker_path), "term": "model_type", "value": type(ml).__name__})
        return pd.DataFrame(rows)

    candidates = sorted(MODELS.glob("*_baseline_metrics.json"))
    if not candidates:
        raise FileNotFoundError(
            f"Missing {stacker_path} and no *_baseline_metrics.json files were found in {MODELS}. "
            "Diagnostics require real coefficient artifacts and will not fabricate them."
        )

    print(f"Missing {stacker_path}; reading meta_coefficients from metrics JSON artifacts.")
    for path in candidates:
        payload = json.loads(path.read_text(encoding="utf-8"))
        coefs = payload.get("meta_coefficients")
        if not isinstance(coefs, dict):
            continue
        print(f"{path.name}: {coefs}")
        horizon = str(payload.get("target", path.stem))
        for term, value in coefs.items():
            rows.append(
                {
                    "source": path.name,
                    "horizon": horizon,
                    "term": term,
                    "value": float(value),
                }
            )
    return pd.DataFrame(rows)


def false_positive_analysis(
    oof_composite: pd.Series,
    y_true: pd.Series,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    At each threshold, count false positives.

    This estimates the false-alarm cost of lowering the threshold to catch
    more pre-recession episodes.
    """
    thresholds = thresholds or THRESHOLDS
    rows = []
    for tau in thresholds:
        pred = (oof_composite >= tau).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        rows.append(
            {
                "threshold": tau,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "false_alarm_months": fp,
            }
        )
    df = pd.DataFrame(rows)
    print("\n=== Threshold vs false positives ===")
    print(df.to_string(index=False))
    df.to_csv(FIGURES / "diagnostic_threshold_fp.csv", index=False)
    return df


def _actual_recession_fold_membership(oof: pd.DataFrame, y_true: pd.Series) -> pd.DataFrame:
    if "fold_idx" not in oof.columns:
        return pd.DataFrame()
    rows = []
    for name, r0 in RECESSION_STARTS.items():
        r0_ts = pd.Timestamp(r0)
        recession_window = y_true.loc[r0_ts : r0_ts + pd.DateOffset(months=11)]
        recession_window = recession_window[recession_window == 1]
        if recession_window.empty:
            rows.append({"scenario": name, "folds": "none", "months": 0})
            continue
        folds = sorted(oof.loc[recession_window.index, "fold_idx"].dropna().astype(int).unique().tolist())
        rows.append({"scenario": name, "folds": ", ".join(map(str, folds)), "months": len(recession_window)})
    return pd.DataFrame(rows)


def _to_markdown_table(df: pd.DataFrame, *, include_index: bool = False) -> str:
    if df.empty:
        return "_No rows._"

    table = df.copy()
    if include_index:
        table = table.reset_index()
    table = table.astype(object).where(pd.notna(table), "")

    headers = [str(col) for col in table.columns]
    rows = [[str(value) for value in row] for row in table.to_numpy()]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows))
        for idx in range(len(headers))
    ]

    def fmt(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([fmt(headers), sep, *(fmt(row) for row in rows)])


def _markdown_summary(
    label_source: str,
    fold_counts: pd.DataFrame,
    peaks: pd.DataFrame,
    sweep: pd.DataFrame,
    coefs: pd.DataFrame,
    fp: pd.DataFrame,
    recession_folds: pd.DataFrame,
) -> str:
    zero_positive_folds = fold_counts.loc[fold_counts["test_pos"] == 0, "fold"].tolist()
    close_peaks = peaks.loc[
        peaks["peak_pre_r0"].notna()
        & (peaks["peak_pre_r0"] < FITTED_THRESHOLD)
        & (peaks["peak_pre_r0"] >= 0.20),
        "scenario",
    ].tolist()
    no_signal = peaks.loc[
        peaks["peak_pre_r0"].notna()
        & (peaks["peak_pre_r0"] < 0.10),
        "scenario",
    ].tolist()
    low_threshold_catches = []
    if 0.10 in sweep.columns and FITTED_THRESHOLD in sweep.columns:
        for scenario in sweep.index:
            if pd.notna(sweep.loc[scenario, 0.10]) and pd.isna(sweep.loc[scenario, FITTED_THRESHOLD]):
                low_threshold_catches.append(scenario)

    coef_lines = []
    if not coefs.empty:
        for _, row in coefs.iterrows():
            horizon = f"{row.get('horizon', '')} " if "horizon" in coefs.columns else ""
            coef_lines.append(f"- {horizon}{row['term']}: {row['value']}")
    else:
        coef_lines.append("- No coefficient rows were available.")

    summary = f"""
# AWRY Post-Fix Diagnostic Summary

Input label source: `{label_source}`

Generated CSV outputs:
- `artifacts/figures/diagnostic_fold_counts.csv`
- `artifacts/figures/diagnostic_peaks.csv`
- `artifacts/figures/diagnostic_threshold_sweep.csv`
- `artifacts/figures/diagnostic_threshold_fp.csv`

## Required Results

### (a) Fold containing recession months
{_to_markdown_table(recession_folds) if not recession_folds.empty else "No `fold_idx` column was available in the OOF artifact."}

### (b) Peak OOF probability in the 12 months before each recession
{_to_markdown_table(peaks)}

### (c) Lead times by threshold
{_to_markdown_table(sweep, include_index=True)}

### (d) Stacker meta-learner coefficients
{chr(10).join(coef_lines)}

### (e) False-positive counts by threshold
{_to_markdown_table(fp)}

## How To Interpret: Case A-E

### Case A: Fold structure artifact
Zero-positive reconstructed test folds: `{zero_positive_folds}`.
If a recession is absent from a test fold, that fold cannot validate recession detection. Use the fold table above to qualify OOS claims.

### Case B: Under-threshold signal
Scenarios with peak probabilities close to the fitted `{FITTED_THRESHOLD:.4f}` threshold: `{close_peaks}`.
If a scenario is listed here, the model almost fired and threshold policy is the main issue.

### Case C: Genuine no-signal episode
Scenarios with pre-R0 peak below 0.10: `{no_signal}`.
If a scenario is listed here, the model did not meaningfully warn before R0 in the OOF series.

### Case D: Threshold tradeoff
Scenarios caught at 0.10 but not at `{FITTED_THRESHOLD:.4f}`: `{low_threshold_catches}`.
If this list is non-empty, report both strict-threshold and recall-oriented lead times rather than changing the model.

### Case E: False-positive cost
Use `diagnostic_threshold_fp.csv` to decide whether a lower threshold is worth the extra false-alarm months. This diagnostic does not recommend changing the threshold; it only quantifies the tradeoff.
"""
    return summary.strip()


if __name__ == "__main__":
    _prepare_diagnostic_dirs()
    composite, y, oof_frame, label_source = _load_oof_predictions_and_labels()

    print(f"\nLoaded OOF composite from {OOF_PREDS / 'composite_oof.parquet'}")
    print(f"Loaded true labels from {label_source}")
    print(f"OOF rows: {len(composite)}; window: {composite.index.min().date()} to {composite.index.max().date()}")

    fold_counts_df = report_fold_positive_counts(y)
    peaks_df = report_pre_recession_probability_peaks(composite, RECESSION_STARTS)
    sweep_df = lead_time_threshold_sweep(composite, RECESSION_STARTS, thresholds=THRESHOLDS)
    coefs_df = inspect_stacker_coefficients(MODELS / "stacker.joblib")
    fp_df = false_positive_analysis(composite, y, thresholds=THRESHOLDS)
    recession_folds_df = _actual_recession_fold_membership(oof_frame, y)

    print(
        "\n"
        + _markdown_summary(
            label_source,
            fold_counts_df,
            peaks_df,
            sweep_df,
            coefs_df,
            fp_df,
            recession_folds_df,
        )
    )

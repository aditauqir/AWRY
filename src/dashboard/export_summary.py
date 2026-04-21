"""Build a Markdown report for download (LLM-friendly, charts as tables)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import FIGURES_DIR, MODELS_DIR
from features import equity_config as _equity_config

# Support older equity_config that only defined DEFAULT_EQUITY_SERIES.
EQUITY_FRED_SERIES_ID = getattr(_equity_config, "EQUITY_FRED_SERIES_ID", _equity_config.DEFAULT_EQUITY_SERIES)
EQUITY_DISPLAY_NAME = getattr(_equity_config, "EQUITY_DISPLAY_NAME", "NASDAQ Composite")


def _fmt_pct(x: float | None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{100.0 * float(x):.2f}%"


def _fmt_float(x: float, nd: int = 4) -> str:
    if isinstance(x, float) and np.isnan(x):
        return "—"
    return f"{float(x):.{nd}f}"


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """GitHub-flavored markdown table without extra dependencies."""
    if df.empty:
        return "_No rows._\n"
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        cells: list[str] = []
        for c in df.columns:
            v = row[c]
            if isinstance(v, (float, np.floating)) and np.isnan(v):
                cells.append("—")
            elif isinstance(v, pd.Timestamp):
                cells.append(v.strftime("%Y-%m-%d"))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _load_alfred_comparison() -> pd.DataFrame:
    path = FIGURES_DIR / "alfred_comparison.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_json_artifact(name: str) -> dict[str, Any]:
    """Return an artifact JSON payload, or an empty dict if missing/unreadable."""
    path = MODELS_DIR / name
    if not path.exists():
        return {}
    try:
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_csv_artifact(name: str) -> tuple[pd.DataFrame, str]:
    """Return a diagnostic CSV plus a note instead of crashing when missing."""
    path = FIGURES_DIR / name
    if not path.exists():
        return pd.DataFrame(), f"File not found: {path.as_posix()}"
    try:
        return pd.read_csv(path), ""
    except Exception as exc:
        return pd.DataFrame(), f"Could not read {path.as_posix()}: {exc}"


def _append_missing_note(df: pd.DataFrame, note: str) -> pd.DataFrame:
    """Create a visible empty table row when a diagnostic artifact is absent."""
    if not note:
        return df
    return pd.DataFrame([{"note": note}])


def _nearest_threshold_column(df: pd.DataFrame, threshold: float) -> str | None:
    """Find the threshold-sweep column closest to the fitted threshold."""
    best_col: str | None = None
    best_gap = float("inf")
    for col in df.columns:
        if col == "scenario":
            continue
        try:
            gap = abs(float(col) - float(threshold))
        except (TypeError, ValueError):
            continue
        if gap < best_gap:
            best_gap = gap
            best_col = str(col)
    return best_col


def _load_diagnostic_tables(fitted_threshold: float) -> dict[str, pd.DataFrame]:
    """Load diagnostic tables used by the report export."""
    fold_counts, fold_note = _load_csv_artifact("diagnostic_fold_counts.csv")
    peaks, peaks_note = _load_csv_artifact("diagnostic_peaks.csv")
    sweep, sweep_note = _load_csv_artifact("diagnostic_threshold_sweep.csv")
    fp, fp_note = _load_csv_artifact("diagnostic_threshold_fp.csv")

    fold_counts = _append_missing_note(fold_counts, fold_note)
    fp = _append_missing_note(fp, fp_note)

    # The threshold sweep may have been generated with a rounded
    # fitted threshold column (for example 0.2325), while thresholds.json stores
    # the exact fitted value. We report the nearest available sweep column and
    # keep that column name visible so the table is traceable to the artifact.
    if peaks_note or sweep_note or peaks.empty or sweep.empty:
        lead_table = pd.DataFrame(
            [
                {
                    "note": "; ".join(note for note in (peaks_note, sweep_note) if note)
                    or "No scenario lead-time rows available.",
                }
            ]
        )
    else:
        threshold_col = _nearest_threshold_column(sweep, fitted_threshold)
        lead_table = peaks.copy()
        if threshold_col is not None:
            lead_cols = sweep[["scenario", threshold_col]].rename(
                columns={threshold_col: "lead_months_at_nearest_threshold"}
            )
            lead_table = lead_table.merge(lead_cols, on="scenario", how="left")
            lead_table["nearest_threshold_in_sweep"] = threshold_col
        else:
            lead_table["lead_months_at_nearest_threshold"] = "not available"
            lead_table["nearest_threshold_in_sweep"] = "not available"

    return {
        "fold_counts": fold_counts,
        "scenario_leads": lead_table,
        "precision_recall_tradeoff": fp,
    }


def collect_awry_export_payload(
    *,
    generated_at: str,
    pipe: Any,
    hist: pd.DataFrame,
    raw_tail: pd.DataFrame,
    latest_ts: pd.Timestamp,
    fc_ts_latest: pd.Timestamp,
    test_ts: pd.Timestamp,
    test_threshold: float,
    diagnostics: dict[str, float],
    forecast_month_end_fn: Any,
    forecast_outlook_label_fn: Any,
    month_label_fn: Any,
    realized_usrec_3m_fn: Any,
) -> dict[str, Any]:
    """Shared tables and scalars for Markdown and LaTeX exports."""
    alpha = float(pipe.alpha)
    last = hist.iloc[-1]
    p_now = float(last["P_now"])
    p_3m = float(last["P_3m"])
    p_awry = float(last["P_AWRY"])
    c_now = alpha * p_now
    c_3m = (1.0 - alpha) * p_3m
    wf_summary = _load_json_artifact("walk_forward_summary_baseline.json")
    threshold_payload = _load_json_artifact("thresholds.json")
    composite_metrics = wf_summary.get("composite_metrics", {}) if isinstance(wf_summary, dict) else {}
    fitted_threshold = (
        (wf_summary.get("thresholds", {}) or {}).get("threshold")
        if isinstance(wf_summary, dict)
        else None
    )
    if fitted_threshold is None and isinstance(threshold_payload, dict):
        fitted_threshold = threshold_payload.get("threshold")
    class_imbalance = threshold_payload.get("class_imbalance") if isinstance(threshold_payload, dict) else None

    ts_key = pd.Timestamp(test_ts)
    row_test = hist.loc[ts_key]
    if isinstance(row_test, pd.DataFrame):
        row_test = row_test.iloc[-1]
    p_now_t = float(row_test["P_now"])
    p_3m_t = float(row_test["P_3m"])
    p_awry_t = float(row_test["P_AWRY"])
    act_now = (
        float(row_test["target_h0"])
        if "target_h0" in row_test.index and pd.notna(row_test.get("target_h0"))
        else float(row_test["USREC"])
    )
    fc_ts_t = forecast_month_end_fn(test_ts, 3)
    act_3m = realized_usrec_3m_fn(pipe, pd.Timestamp(test_ts))

    snap = pd.DataFrame(
        [
            {"field": "As-of month (t)", "value": month_label_fn(latest_ts)},
            {"field": "Forecast month (t+3)", "value": month_label_fn(fc_ts_latest)},
            {"field": "P_AWRY", "value": _fmt_pct(p_awry)},
            {"field": "P_now", "value": _fmt_pct(p_now)},
            {"field": "P_3m", "value": _fmt_pct(p_3m)},
            {"field": "α·P_now", "value": _fmt_pct(c_now)},
            {"field": "(1−α)·P_3m", "value": _fmt_pct(c_3m)},
            {"field": "NBER USREC (same month)", "value": str(int(last["USREC"]))},
            # These fields are sourced from the current artifact JSONs
            # so the export remains aligned with the fitted model run.
            {
                "field": "Composite AUROC",
                "value": _fmt_float(composite_metrics.get("auroc", diagnostics.get("auroc", float("nan"))), 4),
            },
            {
                "field": "Composite Brier",
                "value": _fmt_float(composite_metrics.get("brier", diagnostics.get("brier", float("nan"))), 4),
            },
            {
                "field": "Composite F1",
                "value": _fmt_float(composite_metrics.get("f1", diagnostics.get("f1", float("nan"))), 4),
            },
            {
                "field": "Fitted threshold tau*",
                "value": _fmt_float(float(fitted_threshold), 6) if fitted_threshold is not None else "â€”",
            },
            {
                "field": "Class imbalance",
                "value": _fmt_float(float(class_imbalance), 6) if class_imbalance is not None else "â€”",
            },
        ]
    )

    h = hist.copy()
    h.insert(0, "month_end", pd.to_datetime(h.index).strftime("%Y-%m-%d"))
    keep = [c for c in ["month_end", "P_AWRY", "P_now", "P_3m", "USREC"] if c in h.columns]
    if "target_h0" in h.columns:
        keep.append("target_h0")
    if "target_h3" in h.columns:
        keep.append("target_h3")
    hist_table = h[keep].copy()
    for c in ("P_AWRY", "P_now", "P_3m"):
        if c in hist_table.columns:
            hist_table[c] = hist_table[c].map(lambda x: _fmt_float(float(x), 6))

    diag = pd.DataFrame(
        [
            {"metric": "AUROC", "value": _fmt_float(diagnostics.get("auroc", float("nan")), 4)},
            {"metric": "Brier", "value": _fmt_float(diagnostics["brier"], 4)},
            {"metric": "F1 @0.5", "value": _fmt_float(diagnostics["f1"], 4)},
        ]
    )

    rt = raw_tail.copy()
    if not rt.empty and isinstance(rt.index, pd.DatetimeIndex):
        rt = rt.reset_index()
        c0 = rt.columns[0]
        rt[c0] = pd.to_datetime(rt[c0]).dt.strftime("%Y-%m-%d")

    test_tbl = pd.DataFrame(
        [
            {"role": "Nowcast (same month)", "P": _fmt_pct(p_now_t), "NBER actual": int(act_now)},
            {
                "role": f"3-month ahead ({month_label_fn(fc_ts_t)})",
                "P": _fmt_pct(p_3m_t),
                "NBER actual": "—" if pd.isna(act_3m) else int(act_3m),
            },
            {"role": "Composite P_AWRY", "P": _fmt_pct(p_awry_t), "NBER actual": "—"},
        ]
    )

    return {
        "generated_at": generated_at,
        "alpha": alpha,
        "w_now_logit": float(pipe.now.w1),
        "w_now_rf": float(pipe.now.w2),
        "w_fc_logit": float(pipe.forecast3.w1),
        "w_fc_rf": float(pipe.forecast3.w2),
        "snap": snap,
        "hist": hist_table,
        "diag": diag,
        "raw_fmt": rt,
        "test_tbl": test_tbl,
        "month_label_test": month_label_fn(test_ts),
        "test_threshold": float(test_threshold),
        "forecast_outlook": forecast_outlook_label_fn(test_ts),
        "p_now_t": p_now_t,
        "p_3m_t": p_3m_t,
        "p_awry_t": p_awry_t,
        "alfred": _load_alfred_comparison(),
        "diagnostic_tables": _load_diagnostic_tables(
            float(fitted_threshold) if fitted_threshold is not None else float(test_threshold)
        ),
    }


def build_awry_markdown_export(
    *,
    generated_at: str,
    pipe: Any,
    hist: pd.DataFrame,
    raw_tail: pd.DataFrame,
    latest_ts: pd.Timestamp,
    fc_ts_latest: pd.Timestamp,
    test_ts: pd.Timestamp,
    test_threshold: float,
    diagnostics: dict[str, float],
    forecast_month_end_fn: Any,
    forecast_outlook_label_fn: Any,
    month_label_fn: Any,
    realized_usrec_3m_fn: Any,
) -> str:
    """Single .md document: metadata, KPIs, full history as tables, test-case block."""
    p = collect_awry_export_payload(
        generated_at=generated_at,
        pipe=pipe,
        hist=hist,
        raw_tail=raw_tail,
        latest_ts=latest_ts,
        fc_ts_latest=fc_ts_latest,
        test_ts=test_ts,
        test_threshold=test_threshold,
        diagnostics=diagnostics,
        forecast_month_end_fn=forecast_month_end_fn,
        forecast_outlook_label_fn=forecast_outlook_label_fn,
        month_label_fn=month_label_fn,
        realized_usrec_3m_fn=realized_usrec_3m_fn,
    )
    alpha = p["alpha"]

    parts: list[str] = []
    parts.append("# AWRY export summary\n")
    parts.append("A summary generated by the AWRY tooling system.\n")
    parts.append(f"- **Generated:** {p['generated_at']}\n")
    parts.append(
        f"- **Equity benchmark:** {EQUITY_DISPLAY_NAME} (FRED `{EQUITY_FRED_SERIES_ID}`); "
        "feature column is `NASDAQCOM`.\n"
    )
    parts.append(
        f"- **Composite:** $P_{{\\mathrm{{AWRY}}}} = \\alpha P_{{\\mathrm{{now}}}} + (1-\\alpha) P_{{\\mathrm{{3m}}}}$ "
        f"with **α = {alpha:g}** (same as `composite_score` in code).\n"
    )
    parts.append(
        f"- **Horizon ensembles (logit + RF):** nowcast — logit {p['w_now_logit']:.4f}, RF {p['w_now_rf']:.4f}; "
        f"3-month — logit {p['w_fc_logit']:.4f}, RF {p['w_fc_rf']:.4f}.\n"
    )
    parts.append("\n## Methodology Notes\n\n")
    parts.append(
        "For the three principal backtest scenarios (2001, 2008, 2020), we supplement the revised-data "
        "evaluation with a point-in-time vintage evaluation using ALFRED (Archival FRED) vintages for "
        "revision-sensitive series — PAYEMS, INDPRO, RRSFS, and W875RX1. Financial and index series "
        "(rates, spreads, VIX, news-based indices) are not revised and use current FRED values. This "
        "produces a model prediction reflecting exactly the information that would have been available to "
        "a real-time analyst on each backtest date.\n"
    )
    parts.append("\n## Latest model row (dashboard KPIs)\n\n")
    parts.append(_dataframe_to_markdown(p["snap"]))

    parts.append("\n## Historical probabilities (tabular — same data as timeline chart)\n\n")
    parts.append(_dataframe_to_markdown(p["hist"]))

    parts.append("\n## Model diagnostics (in-sample, composite vs NBER)\n\n")
    parts.append(_dataframe_to_markdown(p["diag"]))
    diagnostic_tables = p["diagnostic_tables"]
    parts.append("\n## Walk-Forward Fold Structure\n\n")
    parts.append(_dataframe_to_markdown(diagnostic_tables["fold_counts"]))
    parts.append("\n## Scenario Lead Times at Fitted Threshold\n\n")
    parts.append(
        "This table combines pre-recession probability peaks with the nearest fitted-threshold "
        "column available in the threshold-sweep diagnostic artifact.\n\n"
    )
    parts.append(_dataframe_to_markdown(diagnostic_tables["scenario_leads"]))
    parts.append("\n## Precision-Recall Tradeoff\n\n")
    parts.append(_dataframe_to_markdown(diagnostic_tables["precision_recall_tradeoff"]))
    if not p["alfred"].empty:
        parts.append("\n## Vintage vs Revised Comparison\n\n")
        parts.append(
            "Table N compares AWRY's composite probability computed on vintage inputs versus revised inputs "
            "for each backtest date. The vintage-data probability is the honest real-time signal; the revised-data "
            "probability reflects what the model would say today knowing all subsequent revisions. The delta "
            "quantifies the look-ahead bias embedded in any evaluation using revised data.\n\n"
        )
        parts.append(_dataframe_to_markdown(p["alfred"]))
    parts.append("\n## Limitations\n\n")
    parts.append(
        "ALFRED vintage data is used for revision-sensitive series (PAYEMS, INDPRO, RRSFS, W875RX1) in the "
        "three principal backtest scenarios. The walk-forward cross-validation on the full sample, however, "
        "uses current revised data for computational efficiency. This means the walk-forward OOS metrics "
        "incorporate a small look-ahead bias from benchmark revisions, quantified separately via the vintage-vs-revised comparison in Table N.\n"
    )

    parts.append("\n## Raw indicator levels (last rows in model panel)\n\n")
    parts.append(_dataframe_to_markdown(p["raw_fmt"]))

    parts.append("\n## Historical test case (matches dashboard controls)\n\n")
    parts.append(f"- **Selected scenario month:** {p['month_label_test']}\n")
    parts.append(f"- **Decision threshold:** {p['test_threshold']:.2f}\n")
    parts.append(f"- **3-month outlook label:** {p['forecast_outlook']}\n")
    parts.append("\n### Probabilities vs NBER\n\n")
    parts.append(_dataframe_to_markdown(p["test_tbl"]))

    parts.append("\n### Composite decomposition (selected month)\n\n")
    parts.append(
        f"- $P_{{\\mathrm{{AWRY}}}} = {alpha:.4f} \\times {p['p_now_t']:.6f} + {1.0 - alpha:.4f} \\times {p['p_3m_t']:.6f} "
        f"= {p['p_awry_t']:.6f}$\n"
    )
    parts.append(
        f"- **α·P_now** = {_fmt_pct(alpha * p['p_now_t'])}; **(1−α)·P_3m** = {_fmt_pct((1.0 - alpha) * p['p_3m_t'])}\n"
    )

    parts.append("\n---\n*End of export.*\n")
    return "".join(parts)

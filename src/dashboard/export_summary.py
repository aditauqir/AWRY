"""Build a Markdown report for download (LLM-friendly, charts as tables)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

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
    parts.append("\n## Latest model row (dashboard KPIs)\n\n")
    parts.append(_dataframe_to_markdown(p["snap"]))

    parts.append("\n## Historical probabilities (tabular — same data as timeline chart)\n\n")
    parts.append(_dataframe_to_markdown(p["hist"]))

    parts.append("\n## Model diagnostics (in-sample, composite vs NBER)\n\n")
    parts.append(_dataframe_to_markdown(p["diag"]))

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

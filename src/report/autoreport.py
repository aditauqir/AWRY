"""Regenerate AWRY markdown/LaTeX reports from existing artifacts only.

This is the non-training equivalent of the dashboard download export.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import MODELS_DIR, OOF_PRED_DIR, RAW_DATA_DIR, REPORTS_DIR
from dashboard.export_latex import build_awry_latex_export
from dashboard.export_summary import build_awry_markdown_export


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _month_label(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%B %Y")


def _forecast_month_end(ts: pd.Timestamp, months: int = 3) -> pd.Timestamp:
    return pd.Timestamp(ts) + pd.DateOffset(months=months)


def _forecast_outlook_label(ts: pd.Timestamp) -> str:
    return f"{_month_label(_forecast_month_end(ts, 3))} outlook"


def _realized_usrec_3m_ahead(_pipe: Any, ts: pd.Timestamp) -> float:
    fc_ts = _forecast_month_end(ts, 3)
    if fc_ts in _pipe.hist.index and "USREC" in _pipe.hist.columns:
        return float(_pipe.hist.loc[fc_ts, "USREC"])
    return float("nan")


def _load_hist() -> pd.DataFrame:
    path = OOF_PRED_DIR / "composite_oof.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing required report input: {path}")
    hist = pd.read_parquet(path)
    if "date" not in hist.columns:
        raise ValueError(f"{path} must contain a date column")
    hist["date"] = pd.to_datetime(hist["date"])
    return hist.sort_values("date").set_index("date")


def _load_raw_tail(columns: list[str], n: int = 12) -> pd.DataFrame:
    frames: list[pd.Series] = []
    for col in columns:
        path = RAW_DATA_DIR / f"{col}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0)
        if df.empty:
            continue
        series = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        series.index = pd.to_datetime(series.index, errors="coerce")
        series = series.dropna()
        series.name = col
        frames.append(series)
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, axis=1).sort_index().ffill()
    return raw.tail(n)


def _make_pipe(hist: pd.DataFrame) -> SimpleNamespace:
    summary = _load_json(MODELS_DIR / "walk_forward_summary_baseline.json")
    now = _load_json(MODELS_DIR / "nowcast_baseline_metrics.json")
    forecast = _load_json(MODELS_DIR / "forecast3_baseline_metrics.json")
    alpha = float(summary.get("alpha", 1.0))
    now_w = now.get("fixed_weights") or [float("nan"), float("nan")]
    fc_w = forecast.get("fixed_weights") or [float("nan"), float("nan")]
    # The export builders only need fitted weights/alpha and the
    # historical USREC series. This avoids running training just to write a
    # report from already-materialized artifacts.
    return SimpleNamespace(
        alpha=alpha,
        hist=hist,
        now=SimpleNamespace(w1=float(now_w[0]), w2=float(now_w[1])),
        forecast3=SimpleNamespace(w1=float(fc_w[0]), w2=float(fc_w[1])),
    )


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    hist = _load_hist()
    pipe = _make_pipe(hist)
    summary = _load_json(MODELS_DIR / "walk_forward_summary_baseline.json")
    diagnostics = summary.get("composite_metrics", {})
    latest_ts = pd.Timestamp(hist.index[-1])
    fc_ts_latest = _forecast_month_end(latest_ts, 3)
    # Match the dashboard's default historical test case near the GFC
    # using artifact data only; no model fitting is performed here.
    test_candidates = [idx for idx in hist.index if pd.Timestamp(idx).strftime("%Y-%m") == "2007-09"]
    test_ts = pd.Timestamp(test_candidates[0] if test_candidates else hist.index[-1])
    threshold = float((summary.get("thresholds") or {}).get("threshold", 0.5))
    raw_tail = _load_raw_tail(
        [
            "PAYEMS",
            "INDPRO",
            "W875RX1",
            "RRSFS",
            "UNRATE",
            "ICSA",
            "T10Y3M",
            "FEDFUNDS",
            "BAA10Y",
            "HOUST",
            "PERMIT",
            "NASDAQCOM",
            "VIXCLS",
        ]
    )

    newest_input = max(
        path.stat().st_mtime
        for path in [
            MODELS_DIR / "walk_forward_summary_baseline.json",
            MODELS_DIR / "thresholds.json",
            OOF_PRED_DIR / "composite_oof.parquet",
        ]
        if path.exists()
    )
    generated_at = datetime.fromtimestamp(newest_input).strftime("%Y-%m-%d %H:%M:%S")
    stamp = datetime.fromtimestamp(newest_input).strftime("%Y%m%d_%H%M")

    kwargs = dict(
        generated_at=generated_at,
        pipe=pipe,
        hist=hist,
        raw_tail=raw_tail,
        latest_ts=latest_ts,
        fc_ts_latest=fc_ts_latest,
        test_ts=test_ts,
        test_threshold=threshold,
        diagnostics=diagnostics,
        forecast_month_end_fn=_forecast_month_end,
        forecast_outlook_label_fn=_forecast_outlook_label,
        month_label_fn=_month_label,
        realized_usrec_3m_fn=_realized_usrec_3m_ahead,
    )
    markdown = build_awry_markdown_export(**kwargs)
    latex = build_awry_latex_export(**kwargs)

    md_path = REPORTS_DIR / f"awry_summary_{stamp}.md"
    tex_path = REPORTS_DIR / f"awry_export_{stamp}.tex"
    md_path.write_text(markdown, encoding="utf-8", newline="\n")
    tex_path.write_text(latex, encoding="utf-8", newline="\n")
    print(f"Markdown report written: {md_path}")
    print(f"LaTeX report written: {tex_path}")
    print(f"alpha={pipe.alpha}")
    print(f"threshold={threshold}")
    print(f"auroc={diagnostics.get('auroc')}")
    print(f"brier={diagnostics.get('brier')}")
    print(f"f1={diagnostics.get('f1')}")
    print(f"now_weights={[pipe.now.w1, pipe.now.w2]}")
    print(f"forecast_weights={[pipe.forecast3.w1, pipe.forecast3.w2]}")


if __name__ == "__main__":
    main()

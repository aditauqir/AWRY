"""Aggregate AWRY artifacts into one deterministic markdown report bundle.

This module is intentionally read-only with respect to model, feature, and
evaluation artifacts. It only writes summary output under artifacts/report_bundle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = PROJECT_ROOT / "artifacts"
MODELS = ARTIFACTS / "models"
FIGURES = ARTIFACTS / "figures"
OOF_PREDS = ARTIFACTS / "oof_preds"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = ARTIFACTS / "report_bundle"
REPORT_PATH = REPORT_DIR / "report_bundle.md"

RECESSION_STARTS = {
    "2001_dotcom": "2001-03-31",
    "2008_gfc": "2007-12-31",
    "2020_covid": "2020-02-29",
}


@dataclass
class SectionResult:
    title: str
    body: str
    populated: bool
    missing_files: list[str] = field(default_factory=list)


def safe_load_json(path: Path) -> tuple[Any | None, str | None]:
    """Load JSON and return an error string instead of raising."""
    if not path.exists():
        return None, f"File not found: {rel(path)}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:  # pragma: no cover - defensive reporting path
        return None, f"Could not read {rel(path)}: {exc}"


def safe_load_csv(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    """Load CSV and return an error string instead of raising."""
    if not path.exists():
        return None, f"File not found: {rel(path)}"
    try:
        return pd.read_csv(path), None
    except Exception as exc:  # pragma: no cover - defensive reporting path
        return None, f"Could not read {rel(path)}: {exc}"


def safe_load_parquet(path: Path) -> tuple[pd.DataFrame | None, str | None]:
    """Load parquet and return an error string instead of raising."""
    if not path.exists():
        return None, f"File not found: {rel(path)}"
    try:
        return pd.read_parquet(path), None
    except Exception as exc:  # pragma: no cover - defensive reporting path
        return None, f"Could not read {rel(path)}: {exc}"


def rel(path: Path) -> str:
    """Return a stable repo-relative path for report text."""
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def fmt_value(value: Any) -> str:
    """Format values compactly and deterministically for markdown tables."""
    if value is None:
        return ""
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        return f"{value:.6g}"
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str] | None = None) -> str:
    """Render a small markdown table without optional tabulate dependency."""
    if not rows:
        return "_No rows available._"
    cols = columns or list(rows[0].keys())
    header = "| " + " | ".join(cols) + " |"
    divider = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, divider]
    for row in rows:
        values = [fmt_value(row.get(col, "")) for col in cols]
        values = [v.replace("\n", " ").replace("|", "\\|") for v in values]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def section_header() -> SectionResult:
    body = "\n".join(
        [
            "# AWRY Report Bundle",
            "",
            "This bundle aggregates the existing AWRY artifacts for paper writing.",
            "It is generated deterministically from files already present in the repository.",
            "",
            "**Read/write scope:** reads existing artifacts and raw cached data; writes only this report under `artifacts/report_bundle/`.",
        ]
    )
    return SectionResult("header", body, True)


def section_feature_set() -> SectionResult:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for path in sorted(MODELS.glob("walk_forward_summary_*.json")):
        payload, err = safe_load_json(path)
        if err:
            missing.append(err)
            continue
        rows.append(
            {
                "file": rel(path),
                "feature_set": payload.get("feature_set"),
                "alpha": payload.get("alpha"),
                "threshold": (payload.get("thresholds") or {}).get("threshold"),
                "composite_auroc": (payload.get("composite_metrics") or {}).get("auroc"),
                "composite_brier": (payload.get("composite_metrics") or {}).get("brier"),
                "composite_f1": (payload.get("composite_metrics") or {}).get("f1"),
            }
        )
    if not rows:
        missing.append(f"File not found: {rel(MODELS / 'walk_forward_summary_*.json')}")
    body = "## Feature Set Summary\n\n" + markdown_table(rows)
    if missing:
        body += "\n\n" + "\n".join(f"- {msg}" for msg in missing)
    return SectionResult("feature set", body, bool(rows), missing)


def section_fitted_params() -> SectionResult:
    missing: list[str] = []
    rows: list[dict[str, Any]] = []

    thresholds, err = safe_load_json(MODELS / "thresholds.json")
    if err:
        missing.append(err)
    elif isinstance(thresholds, dict):
        rows.extend({"source": "thresholds.json", "parameter": k, "value": v} for k, v in sorted(thresholds.items()))

    alpha_tuned, err = safe_load_json(MODELS / "alpha_tuned.json")
    if err:
        missing.append(err)
    elif isinstance(alpha_tuned, dict):
        rows.extend({"source": "alpha_tuned.json", "parameter": k, "value": v} for k, v in sorted(alpha_tuned.items()))

    for path in sorted(MODELS.glob("*_metrics.json")):
        payload, err = safe_load_json(path)
        if err or not isinstance(payload, dict):
            if err:
                missing.append(err)
            continue
        coefs = payload.get("meta_coefficients")
        if isinstance(coefs, dict):
            for key, value in sorted(coefs.items()):
                rows.append({"source": path.name, "parameter": f"meta_{key}", "value": value})
        if "fixed_weights" in payload:
            rows.append({"source": path.name, "parameter": "fixed_weights", "value": payload.get("fixed_weights")})
        if "alpha" in payload:
            rows.append({"source": path.name, "parameter": "alpha", "value": payload.get("alpha")})

    body = "## Fitted Parameters: Threshold, Alpha, Stacker\n\n" + markdown_table(rows)
    if missing:
        body += "\n\n" + "\n".join(f"- {msg}" for msg in missing)
    return SectionResult("fitted params", body, bool(rows), missing)


def section_oos_metrics() -> SectionResult:
    rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for path in sorted(MODELS.glob("walk_forward_summary_*.json")):
        payload, err = safe_load_json(path)
        if err or not isinstance(payload, dict):
            if err:
                missing.append(err)
            continue
        feature_set = payload.get("feature_set", path.stem.replace("walk_forward_summary_", ""))
        for label in ("now_metrics", "forecast_metrics", "composite_metrics", "in_sample_metrics"):
            metrics = payload.get(label) or {}
            rows.append(
                {
                    "feature_set": feature_set,
                    "metric_block": label,
                    "auroc": metrics.get("auroc"),
                    "brier": metrics.get("brier"),
                    "f1": metrics.get("f1"),
                }
            )

    for path in sorted(MODELS.glob("*_metrics.json")):
        payload, err = safe_load_json(path)
        if err or not isinstance(payload, dict):
            if err:
                missing.append(err)
            continue
        for item in payload.get("fold_metrics", []) or []:
            row = {"source": path.name}
            row.update(item)
            fold_rows.append(row)

    body = "## OOS Metrics\n\n### Composite and Horizon Summaries\n\n"
    body += markdown_table(rows)
    body += "\n\n### Per-Fold Metrics When Available\n\n"
    body += markdown_table(fold_rows[:80]) if fold_rows else "_No per-fold metric rows found in metrics JSON artifacts._"
    return SectionResult("OOS metrics", body, bool(rows), missing)


def section_ablation() -> SectionResult:
    path = MODELS / "ablation_summary.json"
    df, err = safe_load_csv(path)
    missing: list[str] = []
    if err:
        payload, json_err = safe_load_json(path)
        if json_err:
            missing.append(json_err)
            body = "## Ablation\n\n" + json_err
            return SectionResult("ablation", body, False, missing)
        df = pd.DataFrame(payload)
    body = "## Ablation\n\n"
    body += markdown_table(df.to_dict("records")) if df is not None and not df.empty else "_No ablation rows found._"
    return SectionResult("ablation", body, df is not None and not df.empty, missing)


def section_diagnostics() -> SectionResult:
    files = [
        ("Fold positive counts", FIGURES / "diagnostic_fold_counts.csv"),
        ("Pre-recession probability peaks", FIGURES / "diagnostic_peaks.csv"),
        ("Threshold sweep", FIGURES / "diagnostic_threshold_sweep.csv"),
        ("Threshold false positives", FIGURES / "diagnostic_threshold_fp.csv"),
    ]
    missing: list[str] = []
    parts = ["## Diagnostics"]
    populated = False
    for title, path in files:
        df, err = safe_load_csv(path)
        parts.append(f"\n### {title}\n")
        if err:
            missing.append(err)
            parts.append(err)
            continue
        populated = True
        parts.append(markdown_table(df.to_dict("records")))
    return SectionResult("diagnostics", "\n".join(parts), populated, missing)


def section_scenario_backtests() -> SectionResult:
    path = OOF_PREDS / "composite_oof.parquet"
    df, err = safe_load_parquet(path)
    missing: list[str] = []
    if err:
        missing.append(err)
        return SectionResult("scenario backtests", f"## Scenario Backtests\n\n{err}", False, missing)
    assert df is not None
    df = df.copy()
    date_col = "date" if "date" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)
    prob_col = "P_AWRY" if "P_AWRY" in df.columns else df.select_dtypes("number").columns[-1]
    rows: list[dict[str, Any]] = []
    for scenario, r0 in RECESSION_STARTS.items():
        r0_ts = pd.Timestamp(r0)
        window = df.loc[r0_ts - pd.DateOffset(months=12) : r0_ts - pd.DateOffset(months=1)]
        if window.empty:
            rows.append({"scenario": scenario, "r0": r0, "peak_pre_r0": None, "peak_month": None, "rows": 0})
            continue
        peak_idx = window[prob_col].astype(float).idxmax()
        rows.append(
            {
                "scenario": scenario,
                "r0": r0,
                "window_start": window.index.min().strftime("%Y-%m-%d"),
                "window_end": window.index.max().strftime("%Y-%m-%d"),
                "peak_pre_r0": float(window.loc[peak_idx, prob_col]),
                "peak_month": peak_idx.strftime("%Y-%m-%d"),
                "rows": len(window),
            }
        )
    body = "## Scenario Backtests From OOF Composite\n\n" + markdown_table(rows)
    return SectionResult("scenario backtests", body, True, missing)


def section_alfred() -> SectionResult:
    path = FIGURES / "alfred_comparison.csv"
    df, err = safe_load_csv(path)
    missing: list[str] = []
    if err:
        missing.append(err)
        return SectionResult("ALFRED vintage comparison", f"## ALFRED Vintage-vs-Revised Comparison\n\n{err}", False, missing)
    assert df is not None
    body = "## ALFRED Vintage-vs-Revised Comparison\n\n" + markdown_table(df.to_dict("records"))
    tex_path = FIGURES / "alfred_comparison.tex"
    if tex_path.exists():
        body += f"\n\nLaTeX companion file found: `{rel(tex_path)}`."
    else:
        missing.append(f"File not found: {rel(tex_path)}")
        body += f"\n\nFile not found: {rel(tex_path)}"
    return SectionResult("ALFRED vintage comparison", body, not df.empty, missing)


def section_current_state() -> SectionResult:
    path = OOF_PREDS / "composite_oof.parquet"
    df, err = safe_load_parquet(path)
    missing: list[str] = []
    if err:
        missing.append(err)
        return SectionResult("current state", f"## Current State: Latest OOF Row\n\n{err}", False, missing)
    assert df is not None
    date_col = "date" if "date" in df.columns else df.columns[0]
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    latest = df.sort_values(date_col).tail(1)
    body = "## Current State: Latest OOF Row\n\n" + markdown_table(latest.to_dict("records"))
    return SectionResult("current state", body, not latest.empty, missing)


def section_recent_raw_features() -> SectionResult:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    if not DATA_RAW.exists():
        missing.append(f"File not found: {rel(DATA_RAW)}")
    for path in sorted(DATA_RAW.glob("*.csv")):
        try:
            df = pd.read_csv(path, index_col=0)
        except Exception as exc:  # pragma: no cover - defensive reporting path
            missing.append(f"Could not read {rel(path)}: {exc}")
            continue
        if df.empty:
            continue
        series = df.iloc[:, 0].dropna()
        if series.empty:
            continue
        latest_date = pd.to_datetime(series.index[-1], errors="coerce")
        rows.append(
            {
                "series": path.stem,
                "latest_date": latest_date.strftime("%Y-%m-%d") if pd.notna(latest_date) else series.index[-1],
                "latest_value": float(series.iloc[-1]),
                "observations": len(series),
            }
        )
    body = "## Recent Raw Feature Values\n\n"
    body += markdown_table(rows, ["series", "latest_date", "latest_value", "observations"])
    if missing:
        body += "\n\n" + "\n".join(f"- {msg}" for msg in missing)
    return SectionResult("recent raw feature values", body, bool(rows), missing)


def section_artifact_inventory() -> SectionResult:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    if not ARTIFACTS.exists():
        missing.append(f"File not found: {rel(ARTIFACTS)}")
    else:
        for path in sorted(p for p in ARTIFACTS.rglob("*") if p.is_file()):
            if REPORT_DIR in path.parents:
                continue
            rows.append(
                {
                    "path": rel(path),
                    "size_bytes": path.stat().st_size,
                    "suffix": path.suffix.lower() or "(none)",
                }
            )
    body = "## Full Artifact File Inventory\n\n" + markdown_table(rows)
    if missing:
        body += "\n\n" + "\n".join(f"- {msg}" for msg in missing)
    return SectionResult("artifact inventory", body, bool(rows), missing)


def section_checklist() -> SectionResult:
    body = "\n".join(
        [
            "## Report-Writing Checklist",
            "",
            "- State which feature set is treated as the primary model in the paper.",
            "- Report the fitted alpha and threshold as learned operating parameters, not user preferences.",
            "- Distinguish OOF walk-forward results from fitted/reference-history probabilities.",
            "- Disclose folds with zero positives when discussing AUROC/F1 stability.",
            "- Discuss threshold sensitivity separately from the fitted operating point.",
            "- Report vintage-vs-revised differences for ALFRED scenarios as data-revision sensitivity.",
            "- Use the artifact inventory to cite exact files used for every table or figure.",
        ]
    )
    return SectionResult("report-writing checklist", body, True)


def build_sections() -> list[SectionResult]:
    """Build sections in the required order."""
    return [
        section_header(),
        section_feature_set(),
        section_fitted_params(),
        section_oos_metrics(),
        section_ablation(),
        section_diagnostics(),
        section_scenario_backtests(),
        section_alfred(),
        section_current_state(),
        section_recent_raw_features(),
        section_artifact_inventory(),
        section_checklist(),
    ]


def render_bundle(sections: list[SectionResult]) -> str:
    """Render the final report with a deterministic section-status appendix."""
    body = "\n\n".join(section.body.rstrip() for section in sections)
    status_rows = [
        {
            "section": section.title,
            "populated": "yes" if section.populated else "no",
            "missing_files": len(section.missing_files),
        }
        for section in sections
    ]
    missing = sorted({msg for section in sections for msg in section.missing_files})
    appendix = "\n\n## Bundle Generation Status\n\n"
    appendix += markdown_table(status_rows, ["section", "populated", "missing_files"])
    if missing:
        appendix += "\n\n### Missing or Unreadable Inputs\n\n" + "\n".join(f"- {msg}" for msg in missing)
    else:
        appendix += "\n\nNo missing inputs were reported by the section builders."
    return body + appendix + "\n"


def main() -> None:
    """Write artifacts/report_bundle/report_bundle.md and print a compact summary."""
    sections = build_sections()
    markdown = render_bundle(sections)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(markdown, encoding="utf-8", newline="\n")

    populated = [section.title for section in sections if section.populated]
    missing_sections = [section.title for section in sections if section.missing_files]
    missing_messages = sorted({msg for section in sections for msg in section.missing_files})
    size = REPORT_PATH.stat().st_size

    print(f"Bundle written: {rel(REPORT_PATH)}")
    print(f"Bundle size: {size} bytes")
    print(f"Sections populated successfully ({len(populated)}/{len(sections)}): {', '.join(populated)}")
    print(
        "Sections with missing files "
        f"({len(missing_sections)}/{len(sections)}): "
        f"{', '.join(missing_sections) if missing_sections else 'none'}"
    )
    if missing_messages:
        print("Missing/unreadable inputs:")
        for msg in missing_messages:
            print(f"- {msg}")


if __name__ == "__main__":
    main()

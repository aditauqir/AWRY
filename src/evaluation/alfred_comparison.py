"""
Run the trained AWRY stacker on vintage and revised feature panels for the
three backtest scenarios and produce a comparison table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from awry_pipeline import fit_awry_pipeline
from config import FIGURES_DIR
from features.dataset_builder import add_lags, engineer_features, feature_matrix_columns
from features.equity_config import DEFAULT_EQUITY_SERIES
from features.dataset_builder import build_raw_monthly_panel
from ingestion.alfred_client import get_vintage_panel
from ingestion.vintage_config import VINTAGE_OPTIONAL, VINTAGE_REQUIRED

BACKTEST_DATES = {
    "2001_dotcom": "2001-02-28",
    "2008_gfc": "2007-11-30",
    "2020_covid": "2020-01-31",
}


def build_vintage_panel_for_scenario(scenario: str) -> pd.DataFrame:
    """
    Build a raw panel as it would have looked on the given backtest date.
    """
    as_of = pd.Timestamp(BACKTEST_DATES[scenario])
    vintage_ids = VINTAGE_REQUIRED + VINTAGE_OPTIONAL
    vintage_df = get_vintage_panel(vintage_ids, as_of)
    current_panel = build_raw_monthly_panel(cached=True, equity_series=DEFAULT_EQUITY_SERIES)
    panel = current_panel.copy()

    for col in vintage_df.columns:
        if col in panel.columns:
            panel[col] = vintage_df[col].reindex(panel.index).combine_first(panel[col])
            panel.loc[panel.index <= as_of, col] = vintage_df[col].reindex(panel.index).loc[panel.index <= as_of]

    return panel.loc[:as_of].copy()


def _panel_last_row(panel: pd.DataFrame, pipe) -> pd.DataFrame:
    features = engineer_features(panel)
    selected = feature_matrix_columns(features, feature_set="full")
    features = add_lags(features, selected, max_lag=2)
    row = features.tail(1).reindex(columns=pipe.x_cols)
    return row


def run_model_on_panel(panel: pd.DataFrame, pipe) -> float:
    """
    Apply the fitted AWRY pipeline to the last row of a custom raw panel.
    """
    last_row = _panel_last_row(panel, pipe)
    p_now = pipe.now.predict_proba(last_row)[0]
    p_3m = pipe.forecast3.predict_proba(last_row)[0]
    return float(pipe.alpha * p_now + (1.0 - pipe.alpha) * p_3m)


def compare_vintage_vs_revised() -> pd.DataFrame:
    """
    Produce the vintage-vs-revised comparison table for the three scenarios.
    """
    pipe = fit_awry_pipeline(cached=True, equity_series=DEFAULT_EQUITY_SERIES, feature_set="full", save_artifacts=False)
    revised_panel = build_raw_monthly_panel(cached=True, equity_series=DEFAULT_EQUITY_SERIES)

    rows: list[dict[str, float | str | None]] = []
    for scenario, as_of in BACKTEST_DATES.items():
        as_of_ts = pd.Timestamp(as_of)
        vintage_panel = build_vintage_panel_for_scenario(scenario)
        revised_slice = revised_panel.loc[:as_of_ts].copy()

        p_vintage = run_model_on_panel(vintage_panel, pipe)
        p_revised = run_model_on_panel(revised_slice, pipe)

        payems_v = vintage_panel["PAYEMS"].dropna().iloc[-1] if "PAYEMS" in vintage_panel else None
        payems_r = revised_slice["PAYEMS"].dropna().iloc[-1] if "PAYEMS" in revised_slice else None
        payems_delta = None if payems_v is None or payems_r is None else float((payems_r - payems_v) * 1000.0)

        rows.append(
            {
                "scenario": scenario,
                "as_of_date": as_of,
                "p_awry_vintage": float(p_vintage),
                "p_awry_revised": float(p_revised),
                "delta_pp": float((p_revised - p_vintage) * 100.0),
                "payems_delta_jobs": payems_delta,
            }
        )

    df = pd.DataFrame(rows)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FIGURES_DIR / "alfred_comparison.csv", index=False)
    (FIGURES_DIR / "alfred_comparison.tex").write_text(df.to_latex(index=False, float_format="%.2f"), encoding="utf-8")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    compare_vintage_vs_revised()

"""Feature-set ablations for AWRY."""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import FIGURES_DIR, MODELS_DIR
from dashboard.components.backtest_chart import AWRY_BACKTEST_SIGNAL_THRESHOLD, lead_months_awry
from evaluation.calibration import calibrate_composite_oof
from evaluation.walk_forward import run_full_walk_forward

ABLATION_FEATURE_SETS = {
    "A": "baseline",
    "B": "stress",
    "C": "full",
    "D": "full_news",
}


def run_ablation_suite(*, cached: bool = True, equity_series: str | None = None) -> pd.DataFrame:
    """Run the baseline/stress/full ablations and save a comparison figure."""
    rows: list[dict[str, float | str | None]] = []
    for label, feature_set in ABLATION_FEATURE_SETS.items():
        result = run_full_walk_forward(cached=cached, equity_series=equity_series, feature_set=feature_set)
        calibrated = calibrate_composite_oof(result["composite_oof"], save_artifacts=(feature_set == "full"))
        lead_hist = result["composite_reference"].copy()
        metrics = result["summary"]["composite_metrics"]
        rows.append(
            {
                "ablation": label,
                "feature_set": feature_set,
                "auroc": float(metrics["auroc"]),
                "brier": float(metrics["brier"]),
                "f1": float(metrics["f1"]),
                "lead_2001": lead_months_awry(lead_hist, "2001 Dot-com", threshold=AWRY_BACKTEST_SIGNAL_THRESHOLD),
                "lead_2008": lead_months_awry(lead_hist, "2008 GFC", threshold=AWRY_BACKTEST_SIGNAL_THRESHOLD),
                "lead_2020": lead_months_awry(lead_hist, "2020 COVID", threshold=AWRY_BACKTEST_SIGNAL_THRESHOLD),
            }
        )

    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    palette = ["#1d4ed8", "#059669", "#dc2626", "#7c3aed"]
    for ax, metric in zip(axes, ["auroc", "brier", "f1"]):
        ax.bar(df["ablation"], df[metric], color=palette[: len(df)])
        ax.set_title(metric.upper())
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ablation_comparison.png", dpi=180)
    plt.close(fig)

    (MODELS_DIR / "ablation_summary.json").write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    return df

"""Calibration helpers for OOF AWRY probabilities."""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

from config import FIGURES_DIR, MODELS_DIR, OOF_PRED_DIR


def calibrate_composite_oof(composite_oof: pd.DataFrame, *, save_artifacts: bool = True) -> pd.DataFrame:
    """Fit isotonic calibration on OOF composite probabilities."""
    y_true = composite_oof["USREC"].values.astype(int)
    y_prob = composite_oof["P_AWRY"].values.astype(float)

    iso = IsotonicRegression(out_of_bounds="clip")
    calibrated = iso.fit_transform(y_prob, y_true)
    out = composite_oof.copy()
    out["P_AWRY_calibrated"] = calibrated

    if save_artifacts:
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=8, strategy="quantile")
        frac_pos_cal, mean_pred_cal = calibration_curve(y_true, calibrated, n_bins=8, strategy="quantile")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], linestyle="--", color="#64748b", label="perfect calibration")
        ax.plot(mean_pred, frac_pos, marker="o", color="#2563eb", label="raw")
        ax.plot(mean_pred_cal, frac_pos_cal, marker="o", color="#dc2626", label="isotonic")
        ax.set_title("Reliability Diagram")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "reliability.png", dpi=180)
        plt.close(fig)

        out.reset_index().rename(columns={"index": "date"}).to_parquet(OOF_PRED_DIR / "composite_oof_calibrated.parquet")
        (MODELS_DIR / "calibration.json").write_text(
            json.dumps({"method": "isotonic", "n_obs": int(len(out))}, indent=2),
            encoding="utf-8",
        )

    return out

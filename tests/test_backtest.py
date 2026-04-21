from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.walk_forward import build_backtest_summary  # noqa: E402


def test_build_backtest_summary_uses_supplied_threshold():
    idx = pd.date_range("2018-06-30", "2020-01-31", freq="ME")
    hist = pd.DataFrame({"P_AWRY": 0.0}, index=idx)
    hist.loc[pd.Timestamp("2019-11-30"), "P_AWRY"] = 0.15

    low_threshold = build_backtest_summary(hist, threshold=0.10)
    high_threshold = build_backtest_summary(hist, threshold=0.20)

    assert low_threshold["2020 COVID"] == 2
    assert high_threshold["2020 COVID"] is None

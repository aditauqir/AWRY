"""Feature transforms."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features.transforms import log_diff  # noqa: E402


def test_log_diff():
    s = pd.Series([100.0, 110.0, 121.0])
    g = log_diff(s)
    assert np.isfinite(g.iloc[1])

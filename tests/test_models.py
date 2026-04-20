"""Ensemble math."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.ensemble import ensemble_predict  # noqa: E402


def test_ensemble_extremes():
    p = ensemble_predict(1.0, 0.0, np.array([0.5]), np.array([0.5]))
    assert abs(p[0] - 0.5) < 1e-6

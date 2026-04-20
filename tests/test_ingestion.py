"""Ingestion smoke tests (requires FRED_API_KEY and network when not cached)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ingestion.fred_client import FredClient  # noqa: E402


def test_fred_client_usrec_smoke():
    c = FredClient()
    s = c.fetch_series_cached("USREC", force_refresh=False)
    assert len(s) > 100
    assert s.name == "USREC"

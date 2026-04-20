"""ALFRED vintage-aware pulls via FRED API realtime_start / realtime_end."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


load_dotenv(_project_root() / ".env")

FRED_OBS_URL = "https://api.stlouisfed.org/fred/series/observations"


def fetch_vintage_series(
    series_id: str,
    as_of: str | date,
    observation_start: str = "1970-01-01",
    observation_end: str | None = None,
    api_key: str | None = None,
) -> pd.Series:
    """
    Observations of `series_id` as known on date `as_of` (YYYY-MM-DD).

    Uses FRED observations endpoint with realtime_start and realtime_end set to `as_of`.
    """
    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise ValueError("FRED_API_KEY missing")

    if isinstance(as_of, date):
        as_of_s = as_of.isoformat()
    else:
        as_of_s = str(as_of)

    params: dict[str, str] = {
        "series_id": series_id,
        "api_key": key,
        "file_type": "json",
        "observation_start": observation_start,
        "realtime_start": as_of_s,
        "realtime_end": as_of_s,
    }
    if observation_end:
        params["observation_end"] = observation_end

    r = requests.get(FRED_OBS_URL, params=params, timeout=120)
    r.raise_for_status()
    data = r.json()
    obs = data.get("observations", [])
    if not obs:
        return pd.Series(dtype=float, name=series_id)

    dates = pd.to_datetime([o["date"] for o in obs])
    vals = []
    for o in obs:
        v = o["value"]
        vals.append(float(v) if v not in ("", ".") else float("nan"))
    s = pd.Series(vals, index=dates, name=series_id)
    return s.sort_index()

"""FRED API wrapper using fredapi. Caches series to data/raw/."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

SERIES_IDS = [
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
    "VIXCLS",
    "USREC",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


load_dotenv(_project_root() / ".env")


def _raw_dir() -> Path:
    d = _project_root() / "data" / "raw"
    d.mkdir(parents=True, exist_ok=True)
    return d


class FredClient:
    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("FRED_API_KEY")
        if not key:
            raise ValueError("FRED_API_KEY missing: set in .env or pass api_key=")
        self._fred = Fred(api_key=key)

    def fetch_series(
        self,
        series_id: str,
        observation_start: str | None = "1970-01-01",
        observation_end: str | None = None,
    ) -> pd.Series:
        """Return a pandas Series indexed by observation date."""
        s = self._fred.get_series(
            series_id,
            observation_start=observation_start,
            observation_end=observation_end,
        )
        s.name = series_id
        return s

    def fetch_series_cached(
        self,
        series_id: str,
        observation_start: str | None = "1970-01-01",
        observation_end: str | None = None,
        force_refresh: bool = False,
        max_cache_age_hours: float = 24.0,
        stale_if_last_obs_days: int = 75,
    ) -> pd.Series:
        """
        Load from CSV unless missing, forced, cache file older than ``max_cache_age_hours``,
        or last observation more than ``stale_if_last_obs_days`` before today (pulls fresh FRED data).
        """
        path = _raw_dir() / f"{series_id}.csv"
        if not force_refresh and path.exists():
            try:
                age_sec = time.time() - path.stat().st_mtime
                stale_file = age_sec > max_cache_age_hours * 3600
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                last = pd.Timestamp(df.index.max()).normalize()
                days_behind = (pd.Timestamp.now().normalize() - last).days
                last_stale = days_behind > stale_if_last_obs_days
                if not stale_file and not last_stale:
                    return df[series_id].squeeze()
            except (OSError, ValueError, KeyError):
                pass

        s = self.fetch_series(series_id, observation_start, observation_end)
        pd.DataFrame({series_id: s}).to_csv(path)
        return s

    def fetch_panel(
        self,
        series_ids: list[str] | None = None,
        observation_start: str | None = "1970-01-01",
        cached: bool = True,
    ) -> pd.DataFrame:
        ids = series_ids or SERIES_IDS
        frames: list[pd.Series] = []
        for sid in ids:
            if cached:
                ser = self.fetch_series_cached(sid, observation_start=observation_start)
            else:
                ser = self.fetch_series(sid, observation_start=observation_start)
            frames.append(ser)
        return pd.concat(frames, axis=1)

"""FRED API wrapper using fredapi. Caches series to data/raw/."""

from __future__ import annotations

import time
from typing import Callable

import pandas as pd
from fredapi import Fred

from config import RAW_DATA_DIR, load_project_env, require_env

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
load_project_env()


def _raw_dir():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DATA_DIR


class FredClient:
    def __init__(
        self,
        api_key: str | None = None,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
    ) -> None:
        load_project_env()
        key = api_key or require_env("FRED_API_KEY")
        self._fred = Fred(api_key=key)
        self._max_retries = max_retries
        self._retry_delay_seconds = retry_delay_seconds

    def _fetch_with_retries(self, loader: Callable[[], pd.Series], series_id: str) -> pd.Series:
        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                return loader()
            except Exception as exc:
                last_error = exc
                if attempt == self._max_retries:
                    break
                time.sleep(self._retry_delay_seconds * attempt)
        msg = f"FRED fetch failed for {series_id}"
        if last_error is not None:
            msg = f"{msg}: {last_error}"
        raise ValueError(msg) from last_error

    def fetch_series(
        self,
        series_id: str,
        observation_start: str | None = "1970-01-01",
        observation_end: str | None = None,
    ) -> pd.Series:
        """Return a pandas Series indexed by observation date."""
        s = self._fetch_with_retries(
            lambda: self._fred.get_series(
                series_id,
                observation_start=observation_start,
                observation_end=observation_end,
            ),
            series_id=series_id,
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
        cached_series: pd.Series | None = None
        if not force_refresh and path.exists():
            try:
                age_sec = time.time() - path.stat().st_mtime
                stale_file = age_sec > max_cache_age_hours * 3600
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                cached_series = df[series_id].squeeze()
                last = pd.Timestamp(df.index.max()).normalize()
                days_behind = (pd.Timestamp.now().normalize() - last).days
                last_stale = days_behind > stale_if_last_obs_days
                if not stale_file and not last_stale:
                    return cached_series
            except (OSError, ValueError, KeyError):
                cached_series = None

        try:
            s = self.fetch_series(series_id, observation_start, observation_end)
        except Exception:
            if cached_series is not None:
                return cached_series
            raise
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

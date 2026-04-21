"""
ALFRED (Archival FRED) vintage data client.

Uses fredapi's ALFRED support to retrieve time series as they were known
on specific historical dates. Essential for real-time backtest evaluation
without look-ahead bias from later data revisions.

Reference: https://github.com/mortada/fredapi
"""

from __future__ import annotations

import hashlib
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from fredapi import Fred

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from config import DATA_DIR, load_project_env, require_env

DEFAULT_CACHE_DIR = DATA_DIR / "alfred"


def _get_fred_client() -> Fred:
    api_key = require_env("FRED_API_KEY")
    return Fred(api_key=api_key)


def _cache_key(series_id: str, as_of_date: str) -> str:
    # Keep cache filenames stable and OS-safe. The hash is a little extra,
    # but it helps if a series ID ever contains awkward characters.
    raw = f"{series_id}__{as_of_date}".encode()
    return hashlib.md5(raw).hexdigest()[:16] + f"_{series_id}_{as_of_date}.parquet"


def _normalize_vintage_payload(payload, series_id: str, as_of_date: str) -> pd.Series:
    if isinstance(payload, pd.Series):
        series = payload.copy()
    elif isinstance(payload, pd.DataFrame):
        frame = payload.copy()
        value_col = "value" if "value" in frame.columns else frame.columns[-1]
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"])
            frame = frame.sort_values("date")
            series = frame.groupby("date")[value_col].last()
        else:
            frame = frame.sort_index()
            series = frame[value_col]
    else:
        raise TypeError(f"Unsupported ALFRED payload type for {series_id}: {type(payload)!r}")

    series.index = pd.to_datetime(series.index)
    series = pd.to_numeric(series, errors="coerce")
    series = series[series.index <= pd.Timestamp(as_of_date)]
    series.name = series_id
    return series.sort_index()


def get_series_vintage(
    series_id: str,
    as_of_date: str | pd.Timestamp,
    cache_dir: Path | None = None,
) -> pd.Series:
    """
    Return the time series for `series_id` as it was known on `as_of_date`.

    Uses fredapi.Fred.get_series_as_of_date() under the hood and caches the
    result to disk keyed by (series_id, as_of_date).
    """
    load_project_env()
    as_of_s = pd.Timestamp(as_of_date).strftime("%Y-%m-%d")
    cache_root = cache_dir or DEFAULT_CACHE_DIR
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / _cache_key(series_id, as_of_s)

    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        return cached["value"].rename(series_id)

    fred = _get_fred_client()
    try:
        # Try the real ALFRED vintage path first.
        payload = fred.get_series_as_of_date(series_id, as_of_s)
        series = _normalize_vintage_payload(payload, series_id, as_of_s)
    except Exception as exc:
        # Some series do not expose ALFRED vintages cleanly, so fall back to the
        # revised series truncated at the requested as-of date.
        print(
            f"[alfred] {series_id}: no ALFRED vintage at {as_of_s} "
            f"({exc}); falling back to revised data truncated by date."
        )
        series = fred.get_series(series_id)
        series.index = pd.to_datetime(series.index)
        series = pd.to_numeric(series, errors="coerce")
        series = series[series.index <= pd.Timestamp(as_of_s)]
        series.name = series_id

    series.to_frame(name="value").to_parquet(cache_path)
    return series


def fetch_vintage_series(
    series_id: str,
    as_of: str | date,
    observation_start: str = "1970-01-01",
    observation_end: str | None = None,
    api_key: str | None = None,
) -> pd.Series:
    """
    Backward-compatible wrapper around `get_series_vintage`.
    """
    del api_key
    series = get_series_vintage(series_id, as_of)
    series = series[series.index >= pd.Timestamp(observation_start)]
    if observation_end is not None:
        series = series[series.index <= pd.Timestamp(observation_end)]
    return series


def get_vintage_panel(
    series_ids: list[str],
    as_of_date: str | pd.Timestamp,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Build a monthly panel with each column pulled at its vintage state.
    """
    frames: dict[str, pd.Series] = {}
    for sid in series_ids:
        s = get_series_vintage(sid, as_of_date, cache_dir=cache_dir)
        monthly = s.resample("ME").last()
        frames[sid] = monthly.rename(sid)
    return pd.DataFrame(frames).sort_index()


if __name__ == "__main__":
    load_project_env()
    payems_vintage = get_series_vintage("PAYEMS", "2007-11-30")
    payems_current = _get_fred_client().get_series("PAYEMS")
    vintage_value = payems_vintage.loc["2007-10":"2007-11"].iloc[-1]
    revised_value = payems_current.loc["2007-11"].iloc[-1]
    diff = (revised_value - vintage_value) * 1000.0
    print("Vintage PAYEMS (2007-11 as known on 2007-11-30):", vintage_value)
    print("Revised PAYEMS (2007-11 as known today):", revised_value)
    print("Difference (jobs):", diff)

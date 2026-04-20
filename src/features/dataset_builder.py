"""Merge FRED panel, daily→monthly for NASDAQCOM/VIX, lags, targets for h=0 and h=3."""

from __future__ import annotations

import pandas as pd

from ingestion.aggregator import (
    align_monthly_to_end_index,
    daily_to_monthly_nasdaqcom_log_returns,
    daily_to_monthly_vix_mean,
)
from ingestion.fred_client import FredClient
from features.equity_config import DEFAULT_EQUITY_SERIES
from features.transforms import log_diff

MONTHLY_FRED = [
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
]

LOG_DIFF_COLS = [
    "PAYEMS",
    "INDPRO",
    "W875RX1",
    "RRSFS",
    "ICSA",
    "HOUST",
    "PERMIT",
]

FEATURE_COLUMNS = [
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
    "NASDAQCOM",
    "VIXCLS",
]


def _month_end_last(s: pd.Series) -> pd.Series:
    s = s.dropna().sort_index().astype(float)
    if s.empty:
        return s
    out = s.resample("ME").last()
    out.name = s.name
    return out


def build_raw_monthly_panel(
    client: FredClient | None = None,
    cached: bool = True,
    equity_series: str | None = None,
) -> pd.DataFrame:
    """Levels at month-end; equity daily → monthly return in column `NASDAQCOM`; VIX daily; USREC monthly."""
    c = client or FredClient()
    fr = not cached
    eq = equity_series or DEFAULT_EQUITY_SERIES

    frames: dict[str, pd.Series] = {}
    for sid in MONTHLY_FRED:
        if cached:
            ser = c.fetch_series_cached(sid, force_refresh=fr)
        else:
            ser = c.fetch_series(sid)
        frames[sid] = _month_end_last(ser)

    sp = c.fetch_series_cached(eq, force_refresh=fr) if cached else c.fetch_series(eq)
    vx = c.fetch_series_cached("VIXCLS", force_refresh=fr) if cached else c.fetch_series("VIXCLS")
    sp_m = align_monthly_to_end_index(daily_to_monthly_nasdaqcom_log_returns(sp))
    vx_m = align_monthly_to_end_index(daily_to_monthly_vix_mean(vx))

    ur = c.fetch_series_cached("USREC", force_refresh=fr) if cached else c.fetch_series("USREC")
    usrec_m = _month_end_last(ur)

    df = pd.DataFrame(frames)
    df["NASDAQCOM"] = sp_m.reindex(df.index)
    df["VIXCLS"] = vx_m.reindex(df.index)
    df["USREC"] = usrec_m.reindex(df.index).ffill()
    df = df.sort_index()
    return _trim_start_and_ffill(df)


def _trim_start_and_ffill(df: pd.DataFrame) -> pd.DataFrame:
    """Drop leading rows before all series have started; forward-fill interior monthly gaps."""
    if df.empty:
        return df
    starts = [df[c].first_valid_index() for c in df.columns if df[c].notna().any()]
    if not starts:
        return df
    start = max(starts)
    out = df.loc[start:].copy()
    return out.ffill()


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    for col in LOG_DIFF_COLS:
        if col in df.columns:
            df[col] = log_diff(df[col])
    return df


def add_lags(df: pd.DataFrame, feature_cols: list[str], max_lag: int = 2) -> pd.DataFrame:
    out = df.copy()
    base = [c for c in feature_cols if c in out.columns]
    for lag in range(1, max_lag + 1):
        for c in base:
            out[f"{c}_lag{lag}"] = out[c].shift(lag)
    return out


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Nowcast label = same month NBER; 3-month label = USREC three months ahead (shift(-3))."""
    out = df.copy()
    if "USREC" not in out.columns:
        return out
    out["target_h0"] = out["USREC"].copy()
    out["target_h3"] = out["USREC"].shift(-3)
    return out


def feature_matrix_columns(df: pd.DataFrame) -> list[str]:
    skip = {"USREC", "target_h0", "target_h3"}
    return [c for c in df.columns if c not in skip and not c.startswith("target_")]


def build_model_table(
    client: FredClient | None = None,
    cached: bool = True,
    max_lag: int = 2,
    equity_series: str | None = None,
) -> pd.DataFrame:
    raw = build_raw_monthly_panel(client=client, cached=cached, equity_series=equity_series)
    feat = engineer_features(raw)
    feat = add_lags(feat, FEATURE_COLUMNS, max_lag=max_lag)
    feat = add_targets(feat)
    xcols = feature_matrix_columns(feat)
    feat = feat.dropna(subset=xcols + ["target_h0", "target_h3"])
    # shift(-3) leaves NaN on the last 3 monthly rows — exclude from training
    feat = feat.loc[feat["target_h3"].notna()].copy()
    return feat

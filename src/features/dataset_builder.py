"""Build raw and modeled monthly panels for AWRY."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from features.equity_config import DEFAULT_EQUITY_SERIES
from features.transforms import log_diff
from ingestion.fred_client import FredClient
from ingestion.news_sentiment_loader import load_news_sentiment_monthly

# These groups are the data story for the demo: core macro variables,
# optional stress indicators, and optional news sentiment get combined into one
# month-end table before any model sees the data.
CORE_MONTHLY_FRED = [
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

MONTHLY_STRESS_FRED = [
    "CSUSHPINSA",
    "CFNAI",
    "UMCSENT",
]

HIGH_FREQ_STRESS_FRED = [
    "BAMLH0A0HYM2",
    "TEDRATE",
    "NFCI",
    "DCOILWTICO",
    "USEPUNEWSINDXM",
]

OPTIONAL_START_SERIES = {
    "BAMLH0A0HYM2",
    "TEDRATE",
}

LOG_DIFF_COLS = [
    "PAYEMS",
    "INDPRO",
    "W875RX1",
    "RRSFS",
    "ICSA",
    "HOUST",
    "PERMIT",
    "CSUSHPINSA",
    "DCOILWTICO",
]

CORE_FEATURE_COLUMNS = [
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

STRESS_FEATURE_COLUMNS = [
    "CSUSHPINSA",
    "BAMLH0A0HYM2",
    "NFCI",
]

FULL_EXTRA_FEATURE_COLUMNS = [
    "TEDRATE",
    "CFNAI",
    "UMCSENT",
    "DCOILWTICO",
    "USEPUNEWSINDXM",
]

TREND_SERIES = ["HOUST", "PERMIT", "PAYEMS", "INDPRO", "RRSFS"]
CHANGE12_SERIES = [
    "UNRATE",
    "T10Y3M",
    "FEDFUNDS",
    "BAA10Y",
    "BAMLH0A0HYM2",
    "TEDRATE",
    "NFCI",
    "CFNAI",
    "UMCSENT",
    "USEPUNEWSINDXM",
]

FEATURE_SET_ALIASES = {
    "baseline": "baseline",
    "stress": "stress",
    "full": "full",
    "full_news": "full_news",
}

NEWS_SENTIMENT_FEATURE_COLUMNS = [
    "NEWS_SENTIMENT_XLSX",
    "NEWS_SENTIMENT_XLSX_ma3",
    "NEWS_SENTIMENT_XLSX_chg12",
]

# Keep the legacy export name because multiple dashboard paths import it.
FEATURE_COLUMNS = CORE_FEATURE_COLUMNS.copy()


def _month_end_resample(s: pd.Series, how: str = "last") -> pd.Series:
    s = s.dropna().sort_index().astype(float)
    if s.empty:
        return s
    if how == "mean":
        out = s.resample("ME").mean()
    else:
        out = s.resample("ME").last()
    out.name = s.name
    return out


def _feature_set_name(feature_set: str) -> str:
    key = feature_set.lower()
    if key not in FEATURE_SET_ALIASES:
        raise ValueError(f"Unknown feature_set={feature_set!r}")
    return FEATURE_SET_ALIASES[key]


def _iter_existing(columns: Iterable[str], df: pd.DataFrame) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for col in columns:
        if col in df.columns and col not in seen:
            out.append(col)
            seen.add(col)
    return out


def _selected_raw_feature_columns(feature_set: str) -> list[str]:
    feature_set = _feature_set_name(feature_set)
    cols = CORE_FEATURE_COLUMNS.copy()
    # Feature sets are additive. Baseline is deliberately small; stress,
    # full, and full_news layer in more signals for ablation comparisons.
    if feature_set in {"stress", "full"}:
        cols.extend(STRESS_FEATURE_COLUMNS)
        cols.extend(["SAHM_GAP", "T10Y3M_INVERSION_DURATION"])
        cols.extend([f"{col}_trend12" for col in TREND_SERIES])
        cols.extend([f"{col}_chg12" for col in CHANGE12_SERIES])
        cols.extend([f"{col}_available" for col in ["BAMLH0A0HYM2", "TEDRATE"]])
    if feature_set == "full":
        cols.extend(FULL_EXTRA_FEATURE_COLUMNS)
        cols.extend([f"{col}_available" for col in FULL_EXTRA_FEATURE_COLUMNS if col == "TEDRATE"])
    if feature_set == "full_news":
        cols.extend(STRESS_FEATURE_COLUMNS)
        cols.extend(FULL_EXTRA_FEATURE_COLUMNS)
        cols.extend(["SAHM_GAP", "T10Y3M_INVERSION_DURATION"])
        cols.extend([f"{col}_trend12" for col in TREND_SERIES])
        cols.extend([f"{col}_chg12" for col in CHANGE12_SERIES])
        cols.extend([f"{col}_available" for col in ["BAMLH0A0HYM2", "TEDRATE"]])
        cols.extend(NEWS_SENTIMENT_FEATURE_COLUMNS)
    return list(dict.fromkeys(cols))


def build_raw_monthly_panel(
    client: FredClient | None = None,
    cached: bool = True,
    equity_series: str | None = None,
) -> pd.DataFrame:
    """Build the raw monthly panel used by evaluation and the dashboard."""
    c = client or FredClient()
    fr = not cached
    eq = equity_series or DEFAULT_EQUITY_SERIES

    frames: dict[str, pd.Series] = {}
    # FRED series arrive at different frequencies. Every series is
    # normalized to month-end before the feature builder creates targets/lags.
    for sid in CORE_MONTHLY_FRED + MONTHLY_STRESS_FRED:
        ser = c.fetch_series_cached(sid, force_refresh=fr) if cached else c.fetch_series(sid)
        frames[sid] = _month_end_resample(ser, how="last")

    for sid in HIGH_FREQ_STRESS_FRED:
        ser = c.fetch_series_cached(sid, force_refresh=fr) if cached else c.fetch_series(sid)
        frames[sid] = _month_end_resample(ser, how="mean")

    sp = c.fetch_series_cached(eq, force_refresh=fr) if cached else c.fetch_series(eq)
    vx = c.fetch_series_cached("VIXCLS", force_refresh=fr) if cached else c.fetch_series("VIXCLS")
    ur = c.fetch_series_cached("USREC", force_refresh=fr) if cached else c.fetch_series("USREC")

    df = pd.DataFrame(frames).sort_index()
    df["NASDAQCOM"] = _month_end_resample(log_diff(sp).rename("NASDAQCOM"), how="mean").reindex(df.index)
    df["VIXCLS"] = _month_end_resample(vx, how="mean").reindex(df.index)
    df["USREC"] = _month_end_resample(ur, how="last").reindex(df.index)
    # The local Excel sentiment file is a daily text-based signal.
    # We align it to month-end means so it enters the model like the other
    # high-frequency market and uncertainty indicators.
    try:
        df["NEWS_SENTIMENT_XLSX"] = load_news_sentiment_monthly().reindex(df.index)
    except FileNotFoundError:
        df["NEWS_SENTIMENT_XLSX"] = pd.Series(index=df.index, dtype=float)
    return _trim_start_and_fill(df)


def _trim_start_and_fill(df: pd.DataFrame) -> pd.DataFrame:
    """Trim to the first fully-covered row, but keep discontinued series as missing later on."""
    if df.empty:
        return df

    starts = [
        df[c].first_valid_index()
        for c in df.columns
        if c not in OPTIONAL_START_SERIES and df[c].notna().any()
    ]
    if not starts:
        return df
    start = max(starts)
    out = df.loc[start:].copy()

    # TEDRATE ended in 2022. Keep post-2022 months missing so later imputation
    # can distinguish "series unavailable" from "stress unchanged."
    skip_ffill = {"TEDRATE"}
    fill_cols = [col for col in out.columns if col not in skip_ffill]
    out[fill_cols] = out[fill_cols].ffill()
    return out


def add_structural_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add trend, 12-month change, Sahm, inversion duration, and availability flags."""
    out = df.copy()

    for col in TREND_SERIES:
        if col in out.columns:
            out[f"{col}_trend12"] = out[col].rolling(12, min_periods=6).mean()

    for col in CHANGE12_SERIES:
        if col in out.columns:
            out[f"{col}_chg12"] = out[col] - out[col].shift(12)

    if "UNRATE" in out.columns:
        u = out["UNRATE"].astype(float)
        sahm = u.rolling(3, min_periods=3).mean() - u.rolling(12, min_periods=12).min()
        out["SAHM_GAP"] = sahm

    if "T10Y3M" in out.columns:
        inverted = (out["T10Y3M"].astype(float) < 0.0).astype(int)
        groups = inverted.eq(0).cumsum()
        out["T10Y3M_INVERSION_DURATION"] = inverted.groupby(groups).cumsum()

    if "NEWS_SENTIMENT_XLSX" in out.columns:
        # The sentiment index can be negative, so we keep it in levels
        # and summarize it with a short moving average plus a 12-month change.
        out["NEWS_SENTIMENT_XLSX_ma3"] = out["NEWS_SENTIMENT_XLSX"].rolling(3, min_periods=2).mean()
        out["NEWS_SENTIMENT_XLSX_chg12"] = out["NEWS_SENTIMENT_XLSX"] - out["NEWS_SENTIMENT_XLSX"].shift(12)

    for col in ["BAMLH0A0HYM2", "TEDRATE"]:
        if col in out.columns:
            out[f"{col}_available"] = out[col].notna().astype(int)

    return out


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply stationarity transforms and then add structured trend features."""
    df = raw.copy()
    for col in LOG_DIFF_COLS:
        if col in df.columns:
            df[col] = log_diff(df[col])
    return add_structural_features(df)


def add_lags(df: pd.DataFrame, feature_cols: list[str], max_lag: int = 2) -> pd.DataFrame:
    out = df.copy()
    base = [c for c in feature_cols if c in out.columns]
    for lag in range(1, max_lag + 1):
        for c in base:
            out[f"{c}_lag{lag}"] = out[c].shift(lag)
    return out


def redundant_lag_columns(X: pd.DataFrame, threshold: float = 0.95) -> list[str]:
    """Identify lag features that are nearly duplicates of their base series."""
    to_drop: list[str] = []
    bases = [c for c in X.columns if not c.endswith(("_lag1", "_lag2"))]
    for base in bases:
        if base not in X.columns:
            continue
        for lag in (f"{base}_lag1", f"{base}_lag2"):
            if lag not in X.columns:
                continue
            corr = X[[base, lag]].corr().iloc[0, 1]
            if pd.notna(corr) and abs(float(corr)) > threshold:
                to_drop.append(lag)
    return sorted(set(to_drop))


def drop_redundant_lags(X: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Drop lag features that are effectively copies of their parent series."""
    to_drop = redundant_lag_columns(X, threshold=threshold)
    if not to_drop:
        return X
    return X.drop(columns=to_drop)


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Nowcast label = same month NBER; 3-month label = USREC three months ahead."""
    out = df.copy()
    if "USREC" not in out.columns:
        return out
    out["target_h0"] = out["USREC"].copy()
    out["target_h3"] = out["USREC"].shift(-3)
    return out


def feature_matrix_columns(
    df: pd.DataFrame,
    feature_set: str = "full",
) -> list[str]:
    """Return the feature columns for the requested ablation set."""
    skip = {"USREC", "target_h0", "target_h3"}
    feature_set = _feature_set_name(feature_set)
    selected = _iter_existing(_selected_raw_feature_columns(feature_set), df)
    lagged = []
    for col in selected:
        for lag in (f"{col}_lag1", f"{col}_lag2"):
            if lag in df.columns:
                lagged.append(lag)
    cols = [c for c in selected + lagged if c in df.columns and c not in skip]
    return cols


def build_model_table(
    client: FredClient | None = None,
    cached: bool = True,
    max_lag: int = 2,
    equity_series: str | None = None,
    feature_set: str = "full",
    lag_corr_threshold: float = 0.95,
) -> pd.DataFrame:
    raw = build_raw_monthly_panel(client=client, cached=cached, equity_series=equity_series)
    feat = engineer_features(raw)
    base_cols = feature_matrix_columns(feat, feature_set=feature_set)
    feat = add_lags(feat, base_cols, max_lag=max_lag)
    feat = add_targets(feat)

    xcols = feature_matrix_columns(feat, feature_set=feature_set)
    feat = feat.dropna(subset=["target_h0", "target_h3"])
    feat = feat.loc[feat["target_h3"].notna()].copy()

    reduced = drop_redundant_lags(feat[xcols], threshold=lag_corr_threshold)
    final_cols = list(reduced.columns)
    keep_cols = final_cols + ["USREC", "target_h0", "target_h3"]
    return feat[keep_cols].copy()

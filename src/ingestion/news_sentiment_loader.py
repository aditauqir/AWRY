"""
Load the local news sentiment spreadsheet and align it to the AWRY panel.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_NEWS_SENTIMENT_PATH = Path(r"c:\Users\adita\Downloads\news_sentiment_data.xlsx")


def load_news_sentiment_daily(path: Path = DEFAULT_NEWS_SENTIMENT_PATH) -> pd.Series:
    """
    Load the daily news sentiment series from the Excel workbook.
    """
    if not path.exists():
        raise FileNotFoundError(f"News sentiment workbook not found at {path}")

    df = pd.read_excel(path, sheet_name="Data")
    required = {"date", "News Sentiment"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {sorted(missing)}")

    out = df.loc[:, ["date", "News Sentiment"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    out["News Sentiment"] = pd.to_numeric(out["News Sentiment"], errors="coerce")
    out = out.dropna(subset=["date", "News Sentiment"]).sort_values("date")
    series = out.set_index("date")["News Sentiment"]
    series.name = "NEWS_SENTIMENT_XLSX"
    return series


def load_news_sentiment_monthly(path: Path = DEFAULT_NEWS_SENTIMENT_PATH) -> pd.Series:
    """
    Convert the daily sentiment series to month-end frequency.
    """
    daily = load_news_sentiment_daily(path=path)
    monthly = daily.resample("ME").mean()
    monthly.name = "NEWS_SENTIMENT_XLSX"
    return monthly

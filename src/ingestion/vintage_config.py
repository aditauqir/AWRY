"""
Classification of FRED series by revision sensitivity.

Revision-sensitive series MUST be pulled via ALFRED for honest real-time
backtesting. Non-revised series can use current FRED values.
"""

from __future__ import annotations

# These series get revised enough that we should always use ALFRED vintages
# for honest real-time backtests.
VINTAGE_REQUIRED = [
    "PAYEMS",
    "INDPRO",
    "RRSFS",
    "W875RX1",
]

# These can be pulled from ALFRED too, but their revisions are usually smaller
# than the core activity series above.
VINTAGE_OPTIONAL = [
    "UNRATE",
    "HOUST",
    "PERMIT",
    "ICSA",
]

# These series are effectively not revised, so current FRED values are the
# right point-in-time values.
NO_REVISIONS = [
    "T10Y3M",
    "FEDFUNDS",
    "BAA10Y",
    "VIXCLS",
    "NASDAQCOM",
    "NFCI",
    "DCOILWTICO",
    "USEPUNEWSINDXM",
    "USREC",
]

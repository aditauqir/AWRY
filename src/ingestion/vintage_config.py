"""
Classification of FRED series by revision sensitivity.

Revision-sensitive series MUST be pulled via ALFRED for honest real-time
backtesting. Non-revised series can use current FRED values.
"""

from __future__ import annotations

# COMMENT: Heavily revised series. Benchmark revisions can move values by
# hundreds of thousands of jobs or multiple percentage points.
VINTAGE_REQUIRED = [
    "PAYEMS",
    "INDPRO",
    "RRSFS",
    "W875RX1",
]

# COMMENT: Lightly revised series. Vintage is technically correct but
# the deltas are smaller than the core activity series above.
VINTAGE_OPTIONAL = [
    "UNRATE",
    "HOUST",
    "PERMIT",
    "ICSA",
]

# COMMENT: These market and rate series are effectively non-revised, so
# current FRED values are the correct point-in-time values.
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

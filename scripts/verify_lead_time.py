"""Compare OOF and fitted-reference AWRY probabilities around the 2001 recession.

This script is diagnostic only. It reads existing parquet artifacts and prints
the side-by-side values needed to understand lead-time discrepancies.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OOF_PATH = ROOT / "artifacts" / "oof_preds" / "composite_oof.parquet"
REFERENCE_PATH = ROOT / "artifacts" / "oof_preds" / "composite_reference_baseline.parquet"


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "date" not in df.columns:
        raise ValueError(f"{path} has no 'date' column")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values("date").set_index("date")


def main() -> None:
    print("=== Lead-time probability source comparison ===")
    print(f"OOF source: {OOF_PATH.relative_to(ROOT)}")
    print(f"Reference source: {REFERENCE_PATH.relative_to(ROOT)}")
    print("Window: 1999-01-01 through 2001-02-28")
    print()

    oof = _load(OOF_PATH)
    reference = _load(REFERENCE_PATH)
    window = pd.date_range("1999-01-31", "2001-02-28", freq="ME")

    rows = []
    for ts in window:
        row = {"date": ts.strftime("%Y-%m-%d")}
        for label, frame in (("oof", oof), ("reference", reference)):
            if ts in frame.index:
                row[f"{label}_P_AWRY"] = frame.loc[ts, "P_AWRY"]
                row[f"{label}_P_now"] = frame.loc[ts, "P_now"]
                row[f"{label}_P_3m"] = frame.loc[ts, "P_3m"]
            else:
                row[f"{label}_P_AWRY"] = pd.NA
                row[f"{label}_P_now"] = pd.NA
                row[f"{label}_P_3m"] = pd.NA
        rows.append(row)

    table = pd.DataFrame(rows)
    for col in table.columns:
        if col != "date":
            table[col] = pd.to_numeric(table[col], errors="coerce").round(6)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

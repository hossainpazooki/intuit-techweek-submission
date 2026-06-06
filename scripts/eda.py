"""Reproducible EDA — the initial pass over the data.

Prints the structural facts that drive the modeling approach:
  - split shapes and the temporal (train=past, val/test=cohort weeks) layout
  - the selective-labels gap (outcomes only for prior-approved loans)
  - default-timing distribution (days_to_default), the basis for Deliverable B
  - structural missingness rates

Usage:
    python scripts/eda.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"


def main() -> None:
    tr = pd.read_csv(DATA / "train.csv", parse_dates=["application_timestamp"])
    va = pd.read_csv(DATA / "validation.csv", parse_dates=["application_timestamp"])
    te = pd.read_csv(DATA / "test.csv", parse_dates=["application_timestamp"])

    print("=== shapes ===")
    for name, d in [("train", tr), ("validation", va), ("test", te)]:
        print(f"  {name:10s} {d.shape}  "
              f"{d.application_timestamp.min().date()} -> {d.application_timestamp.max().date()}")
    print(f"  decision set (val+test) = {len(va) + len(te)}  (expect 13306)")

    print("\n=== selective labels (train) ===")
    print(f"  prior-approved (have outcomes): {tr.default_flag.notna().sum()}")
    print(f"  prior-declined  (no outcomes):  {tr.default_flag.isna().sum()}")
    approved = tr[tr.default_flag.notna()]
    print(f"  default rate among approved:    {approved.default_flag.mean():.3f}")

    print("\n=== default timing (defaulted train loans) ===")
    print(tr.loc[tr.default_flag == 1, "days_to_default"].describe().round(1).to_string())

    print("\n=== structural missingness (train) ===")
    for c in ["observed_monthly_revenue_avg_3mo", "aggregate_credit_utilization",
              "prior_approved_amount", "days_since_last_external_decline"]:
        print(f"  {c:36s} {tr[c].isna().mean():.1%} null")


if __name__ == "__main__":
    main()

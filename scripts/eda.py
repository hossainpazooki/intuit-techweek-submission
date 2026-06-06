"""Reproducible EDA — validates the empirical premises the methodology rests on.

This script is the repo's observability/validation layer. Every claim in
METHODOLOGY.md tagged [VALIDATED] is checked here, so the writeup's premises are
backed by runnable code rather than assertion:

  - split shapes and the temporal (train=past, val/test=cohort weeks) layout
  - the selective-labels gap (outcomes only for funded, matured loans)
  - the *deterministic* legacy funding rule (threshold on prior_underwriter_score)
  - default AND payoff timing (payoff at day 60; default point mass at day 90)
  - recovery distribution on defaulters
  - informative missingness (bank-feed block gated by has_linked_bank_feed)
  - intervention-query structure (Deliverable C)
  - selection / overlap (positivity) diagnostics
  - cohort sizes for Deliverable B

Usage:
    python scripts/eda.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"


def _cohort_week(ts: pd.Series, defs: pd.DataFrame) -> pd.Series:
    """Map application_timestamp -> cohort_week (1..13) via the calendar file."""
    week = pd.Series(np.nan, index=ts.index)
    for _, row in defs.iterrows():
        start = pd.Timestamp(row["start_date"])
        end = pd.Timestamp(row["end_date"]) + pd.Timedelta(days=1)
        week[(ts >= start) & (ts < end)] = int(row["cohort_week"])
    return week


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
    print(f"  funded (have outcomes):   {tr.default_flag.notna().sum()}  "
          f"({tr.default_flag.notna().mean():.3f})")
    print(f"  declined (no outcomes):   {tr.default_flag.isna().sum()}")
    approved = tr[tr.default_flag.notna()]
    print(f"  default rate among funded: {approved.default_flag.mean():.4f}  (expect ~0.1745)")
    print(f"  prior_decision in funded set unique: {sorted(approved.prior_decision.unique())}")

    print("\n=== deterministic legacy funding rule ===")
    # prior_decision == 1 iff prior_underwriter_score >= threshold, with no overlap.
    appr = tr.loc[tr.prior_decision == 1, "prior_underwriter_score"]
    decl = tr.loc[tr.prior_decision == 0, "prior_underwriter_score"]
    thr = (appr.min() + decl.max()) / 2
    pred = (tr.prior_underwriter_score >= thr).astype(int)
    print(f"  implied threshold ~ {thr:.4f}")
    print(f"  mismatches (rule vs prior_decision): {int((pred != tr.prior_decision).sum())} / {len(tr)}")
    print(f"  max declined score {decl.max():.5f} < min approved score {appr.min():.5f}  "
          f"(gap -> no overlap -> positivity fails globally)")

    print("\n=== timing: payoff vs default (funded+matured train) ===")
    paid = tr.loc[tr.repayment_status == "paid_in_full", "days_to_full_repayment"]
    dd = tr.loc[tr.default_flag == 1, "days_to_default"]
    print(f"  paid_in_full repay day: min={paid.min():.0f} max={paid.max():.0f}  (expect exactly 60)")
    print(f"  days_to_default: min={dd.min():.0f} median={dd.median():.0f} max={dd.max():.0f}")
    print(f"  defaults in open interval (60,90): {int(((dd > 60) & (dd < 90)).sum())}  (expect 0)")
    print(f"  defaults at exactly day 90:        {int((dd == 90).sum())}  "
          f"({(dd == 90).mean():.3f} of defaults)")

    print("\n=== recovery on defaulters (final_recovered_amount / requested_amount) ===")
    rec = (tr.loc[tr.default_flag == 1, "final_recovered_amount"]
           / tr.loc[tr.default_flag == 1, "requested_amount"])
    print(f"  mean={rec.mean():.4f}  median={rec.median():.4f}  zero-frac={(rec == 0).mean():.4f}")

    print("\n=== informative missingness: bank-feed gating ===")
    bank_cols = ["observed_monthly_revenue_avg_3mo", "observed_revenue_trend_3mo",
                 "observed_revenue_volatility", "observed_cash_balance_p10",
                 "observed_overdraft_count_3mo", "payroll_regularity_score"]
    no_feed = tr[~tr.has_linked_bank_feed.astype(bool)]
    has_feed = tr[tr.has_linked_bank_feed.astype(bool)]
    all_null_when_off = bool(no_feed[bank_cols].isna().all().all())
    none_null_when_on = bool(has_feed[bank_cols].notna().all().all())
    print(f"  has_linked_bank_feed==False fraction: {(~tr.has_linked_bank_feed.astype(bool)).mean():.3f}")
    print(f"  bank-feed block all-null when feed OFF: {all_null_when_off}")
    print(f"  bank-feed block all-present when feed ON: {none_null_when_on}  (=> null IFF feed off)")
    for c in ["days_since_last_external_decline", "days_since_last_inquiry_elsewhere"]:
        print(f"  {c:36s} {tr[c].isna().mean():.1%} null  (null = no prior event)")

    print("\n=== intervention queries (Deliverable C) ===")
    q = pd.read_csv(DATA / "intervention_queries.csv")
    dd_dict = pd.read_csv(DATA / "data_dictionary.csv")
    non_interv = set(dd_dict.loc[dd_dict.intervenable == False, "field"])  # noqa: E712
    n_struct = q.feature_name.isin(non_interv).sum()
    print(f"  queries={len(q)}  applicants={q.applicant_id.nunique()}  features={q.feature_name.nunique()}")
    print(f"  queries on intervenable=False features: {n_struct} ({n_struct / len(q):.3f})")

    print("\n=== overlap / selection (funded vs declined means) ===")
    for c in ["aggregate_credit_utilization", "requested_amount", "vintage_years"]:
        print(f"  {c:30s} funded={tr.loc[tr.prior_decision==1, c].mean():.3f}  "
              f"declined={tr.loc[tr.prior_decision==0, c].mean():.3f}")

    print("\n=== cohort sizes (val+test, Deliverable B) ===")
    defs = pd.read_csv(DATA / "cohort_week_definitions.csv")
    dec = pd.concat([va, te], ignore_index=True)
    cw = _cohort_week(dec.application_timestamp, defs)
    counts = cw.value_counts().sort_index()
    print("  " + "  ".join(f"w{int(k)}={int(v)}" for k, v in counts.items()))
    print(f"  total assigned = {int(counts.sum())}  unassigned = {int(cw.isna().sum())}")


if __name__ == "__main__":
    main()

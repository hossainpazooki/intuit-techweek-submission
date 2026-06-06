"""Data loading, label construction, and split bookkeeping.

The central data fact of this challenge is **selective labels**: outcome fields
(default_flag, days_to_default, repayment_status, ...) are populated ONLY for
loans the prior underwriter approved and that have since matured. Declined and
immature applications have blank outcomes. Any honest model has to reason about
the applicants we never got to observe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

# Bank-feed block: these columns are null IFF has_linked_bank_feed == False.
BANK_FEED_COLUMNS = [
    "observed_monthly_revenue_avg_3mo",
    "observed_revenue_trend_3mo",
    "observed_revenue_volatility",
    "observed_cash_balance_p10",
    "observed_overdraft_count_3mo",
    "payroll_regularity_score",
]

# Structurally-missing columns to flag in addition to the bank-feed block.
MISSINGNESS_FLAG_COLUMNS = BANK_FEED_COLUMNS + [
    "days_since_last_external_decline",
    "days_since_last_inquiry_elsewhere",
]

OUTCOME_COLUMNS = [
    "default_flag",
    "days_to_default",
    "days_to_full_repayment",
    "repayment_status",
    "final_recovered_amount",
    "observation_status",
]

# Columns that are populated as a *consequence* of the prior decision — using
# them as plain features leaks the very selection we're trying to correct for.
# Handle with care (see features.py / causal.py).
PRIOR_DECISION_LEAKAGE_COLUMNS = [
    "prior_decision",
    "prior_approved_amount",
    # prior_underwriter_score is borderline: predictive but a selection proxy.
]


@dataclass
class Dataset:
    """The three splits plus convenience views."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    @property
    def labeled_train(self) -> pd.DataFrame:
        """Prior-approved + matured train rows (the only ones with outcomes)."""
        return self.train[self.train["default_flag"].notna()].copy()

    @property
    def decision_frame(self) -> pd.DataFrame:
        """validation + test = the 13,306 applicants we must decide on (A)."""
        return pd.concat([self.validation, self.test], ignore_index=True)


def load_dataset() -> Dataset:
    """Read the three CSVs with consistent dtypes/parsing.

    - parse application_timestamp as datetime
    - coerce default_flag to float (NaN preserved on unlabeled rows)
    - coerce has_linked_bank_feed to boolean
    - assert shapes (85340 / 4489 / 8817) and the 13,306 decision count
    """
    parse_dates = ["application_timestamp"]
    train = pd.read_csv(config.TRAIN_CSV, parse_dates=parse_dates)
    validation = pd.read_csv(config.VALIDATION_CSV, parse_dates=parse_dates)
    test = pd.read_csv(config.TEST_CSV, parse_dates=parse_dates)

    for df in (train, validation, test):
        if "default_flag" in df.columns:
            df["default_flag"] = pd.to_numeric(df["default_flag"], errors="coerce").astype(float)
        if "has_linked_bank_feed" in df.columns:
            df["has_linked_bank_feed"] = _coerce_bool(df["has_linked_bank_feed"])

    assert len(train) == 85_340, f"train rows {len(train)} != 85340"
    assert len(validation) == 4_489, f"validation rows {len(validation)} != 4489"
    assert len(test) == 8_817, f"test rows {len(test)} != 8817"

    dataset = Dataset(train=train, validation=validation, test=test)
    assert (
        len(dataset.decision_frame) == config.N_DECISION_APPLICANTS
    ), f"decision_frame {len(dataset.decision_frame)} != {config.N_DECISION_APPLICANTS}"
    return dataset


def _coerce_bool(series: pd.Series) -> pd.Series:
    """Normalize a bool-ish column (True/False/'True'/'False'/1/0) to boolean."""
    if series.dtype == bool:
        return series
    mapping = {
        True: True,
        False: False,
        "True": True,
        "False": False,
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        1: True,
        0: False,
    }
    return series.map(lambda v: mapping.get(v, bool(v) if pd.notna(v) else False)).astype(bool)


def assign_cohort_week(df: pd.DataFrame) -> pd.Series:
    """Map application_timestamp -> cohort_week (1..13) via the calendar file.

    Inclusive date ranges (end_date inclusive). Returns float Series; NaN for
    timestamps outside the 13-week window (i.e. train rows).
    """
    defs = pd.read_csv(
        config.COHORT_WEEK_DEFINITIONS_CSV, parse_dates=["start_date", "end_date"]
    )
    ts = pd.to_datetime(df["application_timestamp"]).dt.normalize()
    result = pd.Series(np.nan, index=df.index, dtype=float)
    for _, row in defs.iterrows():
        start = row["start_date"].normalize()
        end = row["end_date"].normalize()
        mask = (ts >= start) & (ts <= end)
        result.loc[mask] = float(row["cohort_week"])
    return result


def build_default_label(df: pd.DataFrame) -> pd.Series:
    """Binary 'defaulted within the 90-day window' target for the PD model (A).

    Source of truth is default_flag; only defined on labeled rows. Returns a
    nullable-Int Series (NaN on unlabeled rows).
    """
    flag = pd.to_numeric(df["default_flag"], errors="coerce")
    return flag.astype("Int64")


def missingness_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a COPY of df with '<col>_was_null' boolean flags added.

    Bank-feed columns are null exactly when has_linked_bank_feed is False;
    days_since_last_external_decline / days_since_last_inquiry_elsewhere are
    null when there was no prior decline / inquiry. The *fact of* missingness
    is itself signal — model it explicitly rather than blindly imputing.
    """
    out = df.copy()
    for col in MISSINGNESS_FLAG_COLUMNS:
        if col in out.columns:
            out[f"{col}_was_null"] = out[col].isna()
        else:
            out[f"{col}_was_null"] = True
    return out

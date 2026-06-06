"""Data loading, label construction, and split bookkeeping.

The central data fact of this challenge is **selective labels**: outcome fields
(default_flag, days_to_default, repayment_status, ...) are populated ONLY for
loans the prior underwriter approved and that have since matured. Declined and
immature applications have blank outcomes. Any honest model has to reason about
the applicants we never got to observe.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import config

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

    TODO:
      - parse application_timestamp as datetime
      - assert shapes (85340 / 4489 / 8817) and the 13,306 decision count
      - normalize bool-ish columns (default_flag, has_linked_bank_feed)
    """
    raise NotImplementedError


def assign_cohort_week(df: pd.DataFrame) -> pd.Series:
    """Map application_timestamp -> cohort_week (1..13) via the calendar file.

    Needed for Deliverable B (which cohort each approved loan belongs to).
    Returns NaN for timestamps outside the 13-week window (i.e. train rows).
    """
    raise NotImplementedError


def build_default_label(df: pd.DataFrame) -> pd.Series:
    """Binary 'defaulted within the 90-day window' target for the PD model (A).

    Source of truth is default_flag; days_to_default drives the survival target
    in survival.py. Only defined on labeled rows.
    """
    raise NotImplementedError


def missingness_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add was-this-null flags for structurally-missing groups.

    Bank-feed columns are null exactly when has_linked_bank_feed is False;
    days_since_last_external_decline is null when there was no prior decline.
    The *fact of* missingness is itself signal — model it explicitly rather
    than blindly imputing.
    """
    raise NotImplementedError

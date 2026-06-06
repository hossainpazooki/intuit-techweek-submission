"""Feature engineering and the intervenable-feature metadata used by C.

The data dictionary marks each field with an ``intervenable`` flag. Deliverable C
only ever intervenes on intervenable features, so we load that mapping here and
expose the feature groups (business_identity, self_reported, bank_feed,
bureau_credit, platform_engagement, application_context, prior_underwriter).

Modeling contract (METHODOLOGY.md):
  - ALL model features are numeric (float). NaN is preserved (HistGB splits on
    missingness); integer-coded categoricals are treated as numeric/ordinal. We
    deliberately avoid pandas 'category' dtype to dodge cross-split alignment.
  - build_features returns a STABLE, SORTED column order so train/decision/cf
    frames line up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Bank-feed block columns: null IFF has_linked_bank_feed == False.
BANK_FEED_COLUMNS = [
    "observed_monthly_revenue_avg_3mo",
    "observed_revenue_trend_3mo",
    "observed_revenue_volatility",
    "observed_cash_balance_p10",
    "observed_overdraft_count_3mo",
    "payroll_regularity_score",
]

# Other structurally-missing columns that get explicit missingness indicators.
EXTRA_MISSINGNESS_COLUMNS = [
    "days_since_last_external_decline",
    "days_since_last_inquiry_elsewhere",
]

MISSINGNESS_COLUMNS = BANK_FEED_COLUMNS + EXTRA_MISSINGNESS_COLUMNS

# Columns excluded from the model matrix: outcomes, selection-leakage, ids, ts.
EXCLUDE_COLS = [
    # outcome columns
    "default_flag",
    "days_to_default",
    "days_to_full_repayment",
    "repayment_status",
    "final_recovered_amount",
    "observation_status",
    # selection leakage / prior-decision consequences
    "prior_decision",
    "prior_approved_amount",
    # The legacy funding rule is a DETERMINISTIC threshold on
    # prior_underwriter_score (funded iff score >= ~0.273), so the labeled
    # (funded) set only ever sees score >= 0.273 while ~44% of the decision
    # population sits BELOW that minimum -- entirely outside training support.
    # Using the score as an outcome feature would (a) extrapolate a legacy-policy
    # artifact for nearly half the decisions and (b) bake the selection signal
    # into the default model. We therefore EXCLUDE it from the outcome model and
    # keep it only for the funding-rule / positivity diagnostic (propensity.py).
    "prior_underwriter_score",
    # identifiers and timestamp
    "business_id",
    "applicant_id",
    "application_timestamp",
]


def load_data_dictionary() -> pd.DataFrame:
    """Read data_dictionary.csv (field, dtype, group, intervenable, notes)."""
    dd = pd.read_csv(config.DATA_DICTIONARY_CSV)
    # Normalize the intervenable flag to a real boolean.
    dd["intervenable"] = (
        dd["intervenable"].astype(str).str.strip().str.lower().map(
            {"true": True, "false": False}
        )
    )
    return dd


def intervenable_features() -> list[str]:
    """Feature names with intervenable=True -- the only valid targets for C."""
    dd = load_data_dictionary()
    return dd.loc[dd["intervenable"] == True, "field"].tolist()  # noqa: E712


def feature_groups() -> dict[str, list[str]]:
    """group name -> list of field names, from the data dictionary."""
    dd = load_data_dictionary()
    groups: dict[str, list[str]] = {}
    for group, sub in dd.groupby("group", sort=False):
        groups[str(group)] = sub["field"].tolist()
    return groups


def _add_missingness_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Append boolean '<col>_was_null' columns for structurally-missing fields.

    Returns a COPY of df with the added columns (mirrors data.missingness_indicators
    but kept local so features.py has no hard runtime dependency on data.py).
    """
    out = df.copy()
    for col in MISSINGNESS_COLUMNS:
        if col in out.columns:
            out[f"{col}_was_null"] = out[col].isna()
        else:
            # Column not present in this frame -> treat as fully null.
            out[f"{col}_was_null"] = True
    return out


def recompute_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute the engineered ratios from current (possibly intervened) RAW cols.

    Operates on RAW-column frames (before build_features) so dag.propagate_intervention
    can refresh descendants after editing an upstream feature. Returns a COPY.
    """
    out = df.copy()

    # debt_to_revenue = existing_debt_obligations / (stated_annual_revenue/12 + 1)
    if "existing_debt_obligations" in out.columns and "stated_annual_revenue" in out.columns:
        out["debt_to_revenue"] = out["existing_debt_obligations"] / (
            out["stated_annual_revenue"] / 12.0 + 1.0
        )

    # prior_default_rate = prior_loans_default_count / (prior_loans_count + 1)
    if "prior_loans_default_count" in out.columns and "prior_loans_count" in out.columns:
        out["prior_default_rate"] = out["prior_loans_default_count"] / (
            out["prior_loans_count"] + 1.0
        )

    # requested_amount_to_observed_revenue: refresh from raw inputs if both present.
    if (
        "requested_amount" in out.columns
        and "observed_monthly_revenue_avg_3mo" in out.columns
    ):
        out["requested_amount_to_observed_revenue"] = (
            out["requested_amount"] / out["observed_monthly_revenue_avg_3mo"]
        )

    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn a raw split into the numeric model matrix.

    - drop EXCLUDE_COLS (outcomes, selection leakage, ids, timestamp)
    - cast has_linked_bank_feed to float (0/1)
    - add missingness indicator columns (float 0/1)
    - add engineered features (debt_to_revenue, prior_default_rate); keep the
      provided requested_amount_to_observed_revenue ratio
    - coerce EVERY column to float (NaN preserved)
    - return with a STABLE, SORTED column order
    """
    work = df.copy()

    # Engineered ratios (computed from raw columns before they are coerced).
    work = recompute_engineered(work)

    # Cast the linked-feed flag to numeric 0/1 (preserve NaN if ever missing).
    if "has_linked_bank_feed" in work.columns:
        work["has_linked_bank_feed"] = work["has_linked_bank_feed"].astype("float64")

    # Add structural missingness indicators (booleans -> floats below).
    work = _add_missingness_indicators(work)

    # Drop excluded columns (ignore any that aren't present in this frame).
    work = work.drop(columns=[c for c in EXCLUDE_COLS if c in work.columns])

    # Coerce ALL remaining columns to float. Non-numeric/bool become float;
    # anything uncoercible -> NaN (preserved). Booleans map to 0.0/1.0.
    for col in work.columns:
        series = work[col]
        if series.dtype == bool:
            work[col] = series.astype("float64")
        else:
            work[col] = pd.to_numeric(series, errors="coerce").astype("float64")

    # Stable, sorted column order.
    work = work.reindex(sorted(work.columns), axis=1)
    return work


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the stable, sorted model-matrix columns for a raw frame."""
    return list(build_features(df).columns)


def align_columns(df: pd.DataFrame, ref_cols: list[str]) -> pd.DataFrame:
    """Reindex a (already-built) feature frame to ref_cols.

    Missing columns are added as NaN; extras are dropped; order matches ref_cols.
    This lets decision/counterfactual frames match the training column layout.
    """
    aligned = df.reindex(columns=list(ref_cols))
    # Keep everything float, NaN preserved.
    return aligned.astype("float64")

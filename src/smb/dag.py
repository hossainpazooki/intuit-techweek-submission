"""Intervention DAG for Deliverable C (counterfactuals).

We model the small set of *mechanical* dependencies between intervenable raw
columns and the engineered descendants computed from them, plus the structural
relationship between ``has_linked_bank_feed`` and the six bank-feed block
columns (which are null IFF the feed is unlinked).

The baseline backdoor adjustment set is empty: we read off interventional PDs
directly from the fitted hazard ensemble after surgically setting the
do(feature=value) value and recomputing the deterministic descendants. This is
sufficient because the hazard model already conditions on the full observed
covariate set, and the engineered ratios are exact functions of their parents.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import features

# --------------------------------------------------------------------------- #
# Structural columns
# --------------------------------------------------------------------------- #

# The six bank-feed block columns: null IFF has_linked_bank_feed == False.
BANK_FEED_BLOCK_COLS = [
    "observed_monthly_revenue_avg_3mo",
    "observed_revenue_trend_3mo",
    "observed_revenue_volatility",
    "observed_cash_balance_p10",
    "observed_overdraft_count_3mo",
    "payroll_regularity_score",
]


def intervenable_dag() -> dict[str, list[str]]:
    """Return the children map (parent -> deterministic descendants).

    Only edges that have a *mechanical* downstream effect we must recompute
    after an intervention are included.
    """
    return {
        "requested_amount": ["requested_amount_to_observed_revenue"],
        "observed_monthly_revenue_avg_3mo": [
            "requested_amount_to_observed_revenue"
        ],
        "existing_debt_obligations": ["debt_to_revenue"],
        "stated_annual_revenue": ["debt_to_revenue"],
        "has_linked_bank_feed": list(BANK_FEED_BLOCK_COLS),
    }


def backdoor_adjustment_set(feature: str) -> list[str]:
    """Baseline: no explicit backdoor adjustment (empty set).

    The hazard ensemble already conditions on the full covariate vector, so we
    read interventional PDs off the model directly after do(feature=value).
    """
    return []


def propagate_intervention(
    raw_row_df: pd.DataFrame,
    feature: str,
    value,
) -> pd.DataFrame:
    """Apply do(feature=value) to a RAW-column frame and recompute descendants.

    Operates on raw columns (before :func:`features.build_features`). Returns a
    COPY with the intervened value set, deterministic engineered descendants
    recomputed, and the bank-feed structural constraint enforced when
    intervening on ``has_linked_bank_feed``.
    """
    out = raw_row_df.copy()

    out[feature] = value

    if feature == "has_linked_bank_feed":
        # Coerce to a truthiness mask over rows.
        if np.isscalar(value) or value is None:
            truthy = bool(value)
            mask = pd.Series(truthy, index=out.index)
        else:
            mask = pd.Series(value, index=out.index).astype(bool)

        for col in BANK_FEED_BLOCK_COLS:
            if col not in out.columns:
                continue
            # Where the feed is unlinked, the block must be NaN.
            out.loc[~mask, col] = np.nan
            # Where the feed is (now) linked but the block is missing, fill with
            # a reasonable central value so the row is well-formed. Use the
            # column median over present values, falling back to 0.0.
            need_fill = mask & out[col].isna()
            if need_fill.any():
                present = out[col].dropna()
                fill_val = float(present.median()) if len(present) else 0.0
                out.loc[need_fill, col] = fill_val

    # Recompute deterministic engineered descendants from current raw columns.
    out = features.recompute_engineered(out)

    return out

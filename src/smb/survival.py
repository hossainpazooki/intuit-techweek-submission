"""Deliverable B — cumulative default trajectory by cohort week x loan age.

This is a survival / discrete-hazard problem, not a classifier. For each
origination cohort week w (1..13) and loan age a weeks (1..13) we predict the
cumulative fraction of *our approved cohort-w loans* that have defaulted by
day 7a.

Key constraints enforced by the validator:
  - exactly the 13x13 = 169 grid
  - cumulative_default_rate in [0, 1]
  - non-decreasing in age within each cohort
  - cdr_lower_90 <= cumulative_default_rate <= cdr_upper_90
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def fit_hazard_model(labeled_train: pd.DataFrame):
    """Fit a discrete-time hazard model from days_to_default.

    Approaches to consider:
      - non-parametric Kaplan-Meier / Nelson-Aalen per relevant stratum
      - discrete-time logistic hazard (loan-age as a covariate) for sharing
        strength across cohorts and conditioning on applicant features
    Remember the cumulative-default curve is over the APPROVED population, so it
    depends on the policy from Deliverable A (which loans we chose to fund).
    """
    raise NotImplementedError


def predict_trajectory(model, decision_frame: pd.DataFrame, decisions: np.ndarray) -> pd.DataFrame:
    """Produce the 169-row B grid.

    Args:
        model: fitted hazard model
        decision_frame: validation+test applicants (need cohort_week here)
        decisions: A's 0/1 approvals, aligned to decision_frame — the cohort
                   curves are conditioned on the loans we actually approve.

    Returns columns exactly:
        cohort_week, loan_age_weeks, cumulative_default_rate,
        cdr_lower_90, cdr_upper_90
    """
    raise NotImplementedError


def enforce_monotone(cdr: np.ndarray) -> np.ndarray:
    """Make a per-cohort age series non-decreasing (cummax) before writing."""
    return np.maximum.accumulate(cdr)

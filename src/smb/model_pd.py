"""Probability-of-default model — the predictive core of Deliverable A.

Two things make this more than a vanilla classifier:

  1. Selective labels. We train on prior-approved loans but must predict on
     everyone (incl. previously-declined applicants). Consider reweighting /
     a selection model / sensitivity analysis rather than ignoring the gap.
  2. Calibration matters as much as ranking — the profit policy and the 90%
     intervals both consume *probabilities*, not just scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def train_pd_model(X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray | None = None):
    """Fit the PD estimator. Returns a fitted model object.

    TODO: pick the learner (gradient boosting is the default workhorse here),
    then wrap it in a calibrator (isotonic / Platt) fit on held-out data.
    """
    raise NotImplementedError


def predict_pd(model, X: pd.DataFrame) -> np.ndarray:
    """Calibrated PD point estimates in [0, 1], one per row."""
    raise NotImplementedError


def selection_adjusted_weights(train_df: pd.DataFrame) -> np.ndarray:
    """Optional: weights that correct prior-approval selection bias.

    e.g. inverse-propensity from a model of prior_decision, so the approved
    sample better represents the full applicant population. Document the
    assumptions (this is the heart of the Deliverable D causal discussion).
    """
    raise NotImplementedError

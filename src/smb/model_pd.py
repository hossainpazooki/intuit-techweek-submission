"""Probability-of-default model — thin adapters for the predictive core of A.

The corrected methodology relies on the weekly competing-risks hazard spine
(see survival.py) for the production PD estimates. This module provides a simple,
self-contained *binary* PD baseline/fallback:

  - a HistGradientBoostingClassifier (NaN-native, all-numeric features) wrapped in
    an isotonic calibrator so the outputs are honest probabilities, not raw scores;
  - selection-bias reweighting that delegates to the propensity diagnostics.

Calibration matters as much as ranking here: the profit policy and the 90%
intervals both consume *probabilities*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from . import config
from . import calibration
from . import propensity


class _CalibratedPD:
    """A fitted HistGB classifier plus an isotonic probability calibrator."""

    def __init__(self, clf: HistGradientBoostingClassifier, iso):
        self.clf = clf
        self.iso = iso

    def _raw_pd(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(dtype=float)
        else:
            X = np.asarray(X, dtype=float)
        proba = self.clf.predict_proba(X)
        classes = list(self.clf.classes_)
        idx = classes.index(1) if 1 in classes else proba.shape[1] - 1
        return proba[:, idx]

    def predict_pd(self, X) -> np.ndarray:
        raw = self._raw_pd(X)
        if self.iso is not None:
            return calibration.apply_isotonic(self.iso, raw)
        return np.clip(raw, 0.0, 1.0)


def train_pd_model(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: np.ndarray | None = None,
) -> _CalibratedPD:
    """Fit the binary PD estimator and wrap it in an isotonic calibrator.

    X is the (already-built) numeric model matrix; y are the binary default
    labels (0/1). HistGB handles NaN natively, so X is passed through as-is.
    The isotonic calibrator is fit on the same data's in-sample scores -- a
    simple, monotone safety net (the production interval machinery lives in the
    hazard ensemble + calibration.py).
    """
    if isinstance(X, pd.DataFrame):
        X_arr = X.to_numpy(dtype=float)
    else:
        X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()

    clf = HistGradientBoostingClassifier(random_state=config.RANDOM_SEED)
    if sample_weight is not None:
        clf.fit(X_arr, y_arr, sample_weight=np.asarray(sample_weight, dtype=float))
    else:
        clf.fit(X_arr, y_arr)

    # Calibrate raw scores -> outcomes (isotonic, monotone, clipped).
    proba = clf.predict_proba(X_arr)
    classes = list(clf.classes_)
    idx = classes.index(1) if 1 in classes else proba.shape[1] - 1
    raw = proba[:, idx]
    iso = None
    try:
        iso = calibration.fit_isotonic(raw, y_arr)
    except Exception:
        iso = None

    return _CalibratedPD(clf, iso)


def predict_pd(model, X: pd.DataFrame) -> np.ndarray:
    """Calibrated PD point estimates in [0, 1], one per row."""
    if hasattr(model, "predict_pd"):
        return np.asarray(model.predict_pd(X), dtype=float)
    # Fallback for a bare estimator.
    if isinstance(X, pd.DataFrame):
        X_arr = X.to_numpy(dtype=float)
    else:
        X_arr = np.asarray(X, dtype=float)
    proba = model.predict_proba(X_arr)
    classes = list(getattr(model, "classes_", [0, 1]))
    idx = classes.index(1) if 1 in classes else proba.shape[1] - 1
    return np.clip(proba[:, idx], 0.0, 1.0)


def selection_adjusted_weights(train_df: pd.DataFrame) -> np.ndarray:
    """Stabilized inverse-propensity weights correcting prior-approval selection.

    Delegates to the propensity diagnostics: fit P(funded), score the training
    rows, then form stabilized IPW for the funded units. By default the main
    pipeline does NOT use these (the hazard spine + conservative upper-bound rule
    handle selection); this is the optional reweighting fallback.
    """
    model = propensity.fit_funding_propensity(train_df)
    X = propensity.features.build_features(train_df)
    e = propensity.propensity_scores(model, X)
    funded_mask = (train_df["prior_decision"] == 1).to_numpy(dtype=bool)
    return propensity.stabilized_ipw(e, funded_mask)

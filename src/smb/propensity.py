"""Funding-propensity diagnostics (selection / positivity honesty checks).

The outcome labels are observed only on *funded* loans (prior_decision == 1).
That makes the labeled set a selected sample. This module fits a model for the
probability of being funded and reports overlap/positivity diagnostics plus
stabilized inverse-propensity weights.

By default the main pipeline does NOT use IPW (the corrected methodology relies
on the hazard spine + conservative upper-bound decision rule); these utilities
exist for diagnostics and an optional reweighting fallback.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from . import config
from . import features


def fit_funding_propensity(train_df: pd.DataFrame) -> HistGradientBoostingClassifier:
    """Fit P(funded) on the model matrix, target = (prior_decision == 1).

    Uses features.build_features so the propensity model sees the same numeric
    matrix as the outcome models. NaN is preserved (HistGB splits on missing).
    """
    X = features.build_features(train_df)
    y = (train_df["prior_decision"] == 1).astype(int).to_numpy()
    clf = HistGradientBoostingClassifier(random_state=config.RANDOM_SEED)
    clf.fit(X.to_numpy(dtype=float), y)
    return clf


def propensity_scores(model: HistGradientBoostingClassifier, X: pd.DataFrame) -> np.ndarray:
    """Return P(funded) for each row of the (already-built) feature matrix X."""
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy(dtype=float)
    else:
        X = np.asarray(X, dtype=float)
    proba = model.predict_proba(X)
    # class column for label 1 (funded); classes_ are sorted [0, 1].
    classes = list(model.classes_)
    idx = classes.index(1) if 1 in classes else proba.shape[1] - 1
    return proba[:, idx]


def stabilized_ipw(e: np.ndarray, funded_mask: np.ndarray) -> np.ndarray:
    """Stabilized inverse-propensity weights for the funded population.

    For funded units: w = P(fund) / e, where P(fund) is the marginal funding
    rate (the stabilizing numerator). Non-funded units get weight 0. Weights are
    clipped to the [1st, 99th] percentile of the funded weights to tame extremes.
    """
    e = np.asarray(e, dtype=float)
    funded_mask = np.asarray(funded_mask, dtype=bool)

    eps = 1e-6
    e_safe = np.clip(e, eps, 1.0 - eps)

    p_fund = float(funded_mask.mean()) if funded_mask.size else 0.0
    w = np.zeros_like(e_safe, dtype=float)
    w[funded_mask] = p_fund / e_safe[funded_mask]

    if funded_mask.any():
        fw = w[funded_mask]
        lo, hi = np.percentile(fw, [1.0, 99.0])
        w[funded_mask] = np.clip(fw, lo, hi)
    return w


def positivity_report(e: np.ndarray) -> dict:
    """Overlap diagnostics for the propensity scores."""
    e = np.asarray(e, dtype=float)
    n = e.size if e.size else 1
    return {
        "min": float(np.min(e)) if e.size else float("nan"),
        "max": float(np.max(e)) if e.size else float("nan"),
        "frac_below_0_01": float(np.mean(e < 0.01)) if e.size else 0.0,
        "frac_above_0_99": float(np.mean(e > 0.99)) if e.size else 0.0,
    }

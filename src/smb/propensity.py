"""Selection / positivity DIAGNOSTICS for the funding rule (no reject inference).

Outcome labels are observed only on *funded* loans (``prior_decision == 1``), so
the labeled set is a selected sample. The natural temptation is reject inference:
fit a funding-propensity model ``e(x) = P(funded | x)`` and reweight the funded
outcomes (IPW) -- or fold a doubly-robust correction into the outcome fit -- to
recover what declined applicants would have done.

**That is not identified here, and this module deliberately does NOT do it.** The
legacy funding decision is a *perfect deterministic threshold* on
``prior_underwriter_score`` (funded iff score >= ~0.273; see
:func:`deterministic_funding_rule`), so the funding propensity is degenerate (0 or
1) and **positivity fails globally**: there are no funded look-alikes for declined
applicants at the same score. Reweighting cannot conjure outcomes the policy never
observed. Declined-region behaviour is reachable only by *extrapolation* (a stated
smoothness assumption), never by IPW / BASL / DR reweighting.

Consequently the outcome model (``survival.py``) is fit **unweighted** on the
funded labeled data, and this module exists purely to *measure and document* the
selection: prove the deterministic rule, and report the positivity diagnostics
that justify abstaining from reweighting. The previous ``stabilized_ipw`` helper
(an inverse-propensity reweighting path) has been removed because using it would
violate the identification assumptions above.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from . import config
from . import features


# --------------------------------------------------------------------------- #
# The standout finding: the legacy funding rule is deterministic
# --------------------------------------------------------------------------- #


def deterministic_funding_rule(
    train_df: pd.DataFrame,
    score_col: str = "prior_underwriter_score",
    decision_col: str = "prior_decision",
) -> dict:
    """Prove (or refute) that funding is a deterministic threshold on the score.

    Reads the RAW ``prior_underwriter_score`` and ``prior_decision`` columns --
    NOT the model matrix, which deliberately excludes the score (see
    ``features.EXCLUDE_COLS``) -- and checks whether ``prior_decision == 1`` is a
    perfect threshold rule on the score.

    Returns a diagnostic dict:
        threshold          midpoint between the max declined and min approved score
        n_mismatches       rows where (score >= threshold) != funded  (0 == perfect)
        max_declined_score largest score among declined rows
        min_approved_score smallest score among approved rows
        gap                min_approved_score - max_declined_score   (>0 == no overlap)
        frac_decisions_below_support
                           among an optional decision frame, the fraction of
                           applicants whose score is below the funded-set minimum
                           (i.e. entirely outside the outcome model's support) --
                           NaN here; computed by :func:`out_of_support_fraction`.
    """
    score = pd.to_numeric(train_df[score_col], errors="coerce")
    funded = _funded_mask(train_df[decision_col])

    max_declined = float(score[~funded].max()) if (~funded).any() else float("nan")
    min_approved = float(score[funded].min()) if funded.any() else float("nan")
    threshold = (max_declined + min_approved) / 2.0
    rule = score >= threshold
    n_mismatches = int((rule.to_numpy() != funded.to_numpy()).sum())

    return {
        "threshold": float(threshold),
        "n_mismatches": n_mismatches,
        "n_rows": int(len(train_df)),
        "max_declined_score": max_declined,
        "min_approved_score": min_approved,
        "gap": float(min_approved - max_declined),
        "frac_decisions_below_support": float("nan"),
    }


def out_of_support_fraction(
    labeled_train: pd.DataFrame,
    decision_frame: pd.DataFrame,
    score_col: str = "prior_underwriter_score",
) -> float:
    """Fraction of decision applicants whose score is below the labeled-set min.

    These applicants sit entirely outside the outcome model's training support on
    the legacy score -- the empirical reason the score is excluded from the
    outcome model and predictions in that region are extrapolation, not estimates.
    """
    lo = pd.to_numeric(labeled_train[score_col], errors="coerce").min()
    dec = pd.to_numeric(decision_frame[score_col], errors="coerce")
    if not np.isfinite(lo) or dec.notna().sum() == 0:
        return float("nan")
    return float((dec < lo).mean())


def _funded_mask(decision: pd.Series) -> pd.Series:
    """Boolean 'was funded' from a prior_decision column (1/'approved'/True)."""
    s = decision.astype(str).str.strip().str.lower()
    return s.isin(["1", "approved", "true"])


# --------------------------------------------------------------------------- #
# Learned funding propensity (secondary positivity diagnostic only)
# --------------------------------------------------------------------------- #


def fit_funding_propensity(train_df: pd.DataFrame) -> HistGradientBoostingClassifier:
    """Fit P(funded) on the OUTCOME model matrix, target = (prior_decision == 1).

    Uses ``features.build_features`` so the propensity model sees the same numeric
    matrix as the outcome model -- which, by design, no longer includes
    ``prior_underwriter_score``. Even without the score, the learned propensity
    concentrates near 0/1 (the rule is near-perfectly recoverable from covariates),
    reinforcing the positivity story. This is a *diagnostic*; its scores are never
    used to reweight the outcome fit.
    """
    X = features.build_features(train_df)
    y = _funded_mask(train_df["prior_decision"]).astype(int).to_numpy()
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
    classes = list(model.classes_)
    idx = classes.index(1) if 1 in classes else proba.shape[1] - 1
    return proba[:, idx]


def positivity_report(e: np.ndarray) -> dict:
    """Overlap diagnostics for the propensity scores (degeneracy => no overlap)."""
    e = np.asarray(e, dtype=float)
    return {
        "min": float(np.min(e)) if e.size else float("nan"),
        "max": float(np.max(e)) if e.size else float("nan"),
        "frac_below_0_01": float(np.mean(e < 0.01)) if e.size else 0.0,
        "frac_above_0_99": float(np.mean(e > 0.99)) if e.size else 0.0,
    }

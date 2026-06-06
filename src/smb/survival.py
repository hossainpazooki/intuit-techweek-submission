"""Deliverable B - cumulative default trajectory via weekly competing-risks hazard.

This is a survival / discrete-hazard problem, not a classifier. We fit ONE
3-class HistGradientBoostingClassifier on a person-period expansion of the
labeled training loans (see survival_data.build_person_periods). Each labeled
loan emits one row per weekly age t=1..T_i with a 3-class target:

    y = 0  survive (no event this week)
    y = 1  default this week
    y = 2  payoff this week

predict_proba columns are ordered by sorted class labels [0,1,2] =
[1 - h_d - h_p, h_d, h_p].

For an applicant x we build 13 synthetic rows (x repeated, loan_age_weeks=1..13),
predict hazards h_d[t], h_p[t], then run the competing-risks survival recursion:

    S[0]      = 1
    S[t]      = prod_{s<=t} (1 - h_d[s] - h_p[s])
    CIF_d[t]  = sum_{s<=t} h_d[s] * S[s-1]
    lifetime_pd = CIF_d[13]

CIF_d is non-decreasing in t by construction.

Validator constraints for the 169-row B grid:
  - exactly the 13x13 = 169 grid
  - cumulative_default_rate in [0, 1]
  - non-decreasing in age within each cohort
  - cdr_lower_90 <= cumulative_default_rate <= cdr_upper_90
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from . import calibration, config, features, survival_data


# --------------------------------------------------------------------------- #
# Hazard model container
# --------------------------------------------------------------------------- #


class HazardModel:
    """A fitted 3-class discrete-time competing-risks hazard estimator.

    Attributes:
        clf: fitted HistGradientBoostingClassifier (classes [0, 1, 2]).
        feature_cols: ordered model columns INCLUDING 'loan_age_weeks'.
    """

    def __init__(self, clf: HistGradientBoostingClassifier, feature_cols: list[str]):
        self.clf = clf
        self.feature_cols = list(feature_cols)


def fit_hazard_model(
    labeled_train: pd.DataFrame,
    sample_weight: np.ndarray | None = None,
    seed: int = config.RANDOM_SEED,
) -> HazardModel:
    """Build person-periods and fit a 3-class HistGB hazard model."""
    X_pp, y_pp = survival_data.build_person_periods(labeled_train)
    feature_cols = list(X_pp.columns)

    clf = HistGradientBoostingClassifier(random_state=seed)
    clf.fit(X_pp.values, y_pp, sample_weight=sample_weight)

    return HazardModel(clf=clf, feature_cols=feature_cols)


# --------------------------------------------------------------------------- #
# Hazard curves + survival recursion
# --------------------------------------------------------------------------- #


def _class_index(clf: HistGradientBoostingClassifier, label: int) -> int | None:
    """Index of `label` in clf.classes_, or None if the class never appeared."""
    classes = list(clf.classes_)
    if label in classes:
        return classes.index(label)
    return None


def hazard_curves(model: HazardModel, X_static: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Predict weekly default/payoff hazards for each applicant.

    Args:
        model: fitted HazardModel.
        X_static: built feature frame WITHOUT 'loan_age_weeks' (it is added here).
            Should already be aligned to model.feature_cols minus 'loan_age_weeks';
            we re-align defensively.

    Returns:
        (h_d, h_p), each shape (N, WEEKS). h_d is the default hazard (class 1),
        h_p the payoff hazard (class 2), per applicant per weekly age 1..WEEKS.
    """
    weeks = config.WEEKS
    n = len(X_static)

    # Static columns the model expects (everything except the age covariate).
    static_cols = [c for c in model.feature_cols if c != "loan_age_weeks"]

    base = features.align_columns(X_static, static_cols)

    idx_d = _class_index(model.clf, 1)
    idx_p = _class_index(model.clf, 2)

    h_d = np.zeros((n, weeks), dtype=float)
    h_p = np.zeros((n, weeks), dtype=float)

    for a in range(1, weeks + 1):
        synth = base.copy()
        synth["loan_age_weeks"] = float(a)
        # Reorder to the exact training column order.
        synth = synth.reindex(columns=model.feature_cols)
        proba = model.clf.predict_proba(synth.values)
        if idx_d is not None:
            h_d[:, a - 1] = proba[:, idx_d]
        if idx_p is not None:
            h_p[:, a - 1] = proba[:, idx_p]

    return h_d, h_p


def cif_default(h_d: np.ndarray, h_p: np.ndarray) -> np.ndarray:
    """Cumulative incidence of default via the competing-risks recursion.

    S[0] = 1; S[t] = prod_{s<=t}(1 - h_d[s] - h_p[s]);
    CIF_d[t] = sum_{s<=t} h_d[s] * S[s-1].

    Args/returns shape (N, WEEKS). CIF_d is non-decreasing in t.
    """
    h_d = np.asarray(h_d, dtype=float)
    h_p = np.asarray(h_p, dtype=float)
    n, weeks = h_d.shape

    # One-step survival probability for each week (clip to a valid range).
    surv_step = np.clip(1.0 - h_d - h_p, 0.0, 1.0)

    cif = np.zeros((n, weeks), dtype=float)
    s_prev = np.ones(n, dtype=float)  # S[s-1], starting at S[0]=1
    acc = np.zeros(n, dtype=float)
    for t in range(weeks):
        acc = acc + h_d[:, t] * s_prev
        cif[:, t] = acc
        s_prev = s_prev * surv_step[:, t]

    return np.clip(cif, 0.0, 1.0)


def lifetime_pd(model: HazardModel, X_static: pd.DataFrame) -> np.ndarray:
    """Lifetime default probability = CIF_d at the final week, per applicant."""
    h_d, h_p = hazard_curves(model, X_static)
    cif = cif_default(h_d, h_p)
    return cif[:, -1]


def default_week_probs(
    model: HazardModel, X_static: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Per-week default probability mass + total payoff probability.

    Decomposes the competing-risks recursion into the marginal probability that
    the loan *defaults in weekly age t* (not just the cumulative CIF):

        P(default in week t) = h_d[t] * S[t-1]
        P(payoff)            = sum_t h_p[t] * S[t-1]

    where S[t-1] = prod_{s<t}(1 - h_d[s] - h_p[s]). This is exactly what
    economics.expected_npv_timing integrates over to price default timing.

    Returns:
        (p_def, p_payoff) with p_def shape (N, WEEKS) and p_payoff shape (N,).
        p_def.sum(axis=1) equals lifetime_pd (CIF_d at the final week).
    """
    h_d, h_p = hazard_curves(model, X_static)
    n, weeks = h_d.shape
    surv_step = np.clip(1.0 - h_d - h_p, 0.0, 1.0)

    p_def = np.zeros((n, weeks), dtype=float)
    p_payoff = np.zeros(n, dtype=float)
    s_prev = np.ones(n, dtype=float)  # S[t-1], starting at S[0]=1
    for t in range(weeks):
        p_def[:, t] = h_d[:, t] * s_prev
        p_payoff += h_p[:, t] * s_prev
        s_prev = s_prev * surv_step[:, t]

    return p_def, p_payoff


# --------------------------------------------------------------------------- #
# Ensemble
# --------------------------------------------------------------------------- #


def fit_hazard_ensemble(
    labeled_train: pd.DataFrame,
    seeds,
) -> list[HazardModel]:
    """Fit a bag of HazardModels, one per seed."""
    return [fit_hazard_model(labeled_train, seed=int(s)) for s in seeds]


# --------------------------------------------------------------------------- #
# Deliverable B trajectory
# --------------------------------------------------------------------------- #


def predict_trajectory(
    models: list[HazardModel],
    decision_frame: pd.DataFrame,
    decisions: np.ndarray,
    cohort_weeks: np.ndarray,
) -> pd.DataFrame:
    """Produce the 169-row B grid (cohort_week x loan_age_weeks).

    For each ensemble member we compute CIF_d for approved applicants
    (decisions == 1), grouped by cohort_week, averaging within each cohort per
    weekly age. Across members we take the mean (point) and 5/95 percentiles
    (bounds), enforce monotonicity in age per cohort, then clip_and_order.

    Empty cohorts fall back to the overall approved-mean curve (never NaN).

    Returns columns exactly:
        cohort_week, loan_age_weeks, cumulative_default_rate,
        cdr_lower_90, cdr_upper_90
    """
    weeks = config.WEEKS
    n_cohorts = config.N_COHORT_WEEKS

    decisions = np.asarray(decisions)
    cohort_weeks = np.asarray(cohort_weeks, dtype=float)

    approved_mask = decisions == 1

    # Build the static feature matrix once (raw -> model matrix). hazard_curves
    # re-aligns to each model's columns and adds loan_age_weeks itself.
    X_full = features.build_features(decision_frame)

    # Per-model, per-cohort CIF curves: stack[m] has shape (n_cohorts, weeks).
    per_model_cohort = np.full((len(models), n_cohorts, weeks), np.nan, dtype=float)
    per_model_overall = np.full((len(models), weeks), np.nan, dtype=float)

    for m, model in enumerate(models):
        h_d, h_p = hazard_curves(model, X_full)
        cif = cif_default(h_d, h_p)  # (N, weeks)

        if approved_mask.any():
            overall = np.nanmean(cif[approved_mask], axis=0)
        else:
            overall = np.nanmean(cif, axis=0)
        per_model_overall[m] = overall

        for c in range(1, n_cohorts + 1):
            sel = approved_mask & (cohort_weeks == float(c))
            if sel.any():
                per_model_cohort[m, c - 1] = np.nanmean(cif[sel], axis=0)
            # else: leave NaN, filled with overall fallback below.

        # Apply overall fallback for empty cohorts (so percentiles are clean).
        for c in range(n_cohorts):
            if np.isnan(per_model_cohort[m, c]).any():
                per_model_cohort[m, c] = overall

    # Aggregate across ensemble members per cohort/age.
    lo_q = config.INTERVAL_LOWER_Q * 100.0
    hi_q = config.INTERVAL_UPPER_Q * 100.0

    rows: list[dict] = []
    for c in range(n_cohorts):
        samples = per_model_cohort[:, c, :]  # (n_models, weeks)
        point = np.mean(samples, axis=0)
        lower = np.percentile(samples, lo_q, axis=0)
        upper = np.percentile(samples, hi_q, axis=0)

        # Monotone (non-decreasing in age) per cohort, on all three series.
        point = enforce_monotone(point)
        lower = enforce_monotone(lower)
        upper = enforce_monotone(upper)

        lower, point, upper = calibration.clip_and_order(lower, point, upper)

        for a in range(weeks):
            rows.append(
                {
                    "cohort_week": c + 1,
                    "loan_age_weeks": a + 1,
                    "cumulative_default_rate": float(point[a]),
                    "cdr_lower_90": float(lower[a]),
                    "cdr_upper_90": float(upper[a]),
                }
            )

    out = pd.DataFrame(
        rows,
        columns=[
            "cohort_week",
            "loan_age_weeks",
            "cumulative_default_rate",
            "cdr_lower_90",
            "cdr_upper_90",
        ],
    )
    return out


def enforce_monotone(cdr: np.ndarray) -> np.ndarray:
    """Make a per-cohort age series non-decreasing (cummax) before writing."""
    return np.maximum.accumulate(cdr)

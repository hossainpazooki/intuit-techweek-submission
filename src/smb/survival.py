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


def fit_cif_scale(
    models: list[HazardModel],
    calib_frame: pd.DataFrame,
    y_calib: np.ndarray,
    lo: float = 0.8,
    hi: float = 1.5,
) -> float:
    """Global CIF recalibration factor = realized lifetime rate / mean predicted
    lifetime PD, fit on an out-of-time funded set with outcomes. Clipped to
    [lo, hi] to stay robust. Corrects the hazard's systematic OOT under-prediction.
    """
    from . import features as _features

    y = np.asarray(y_calib, dtype=float)
    X = _features.build_features(calib_frame)
    preds = []
    for m in models:
        sc = [c for c in m.feature_cols if c != "loan_age_weeks"]
        preds.append(np.asarray(lifetime_pd(m, _features.align_columns(X, sc)), dtype=float))
    pred_mean = float(np.mean(preds))
    if pred_mean <= 1e-9:
        return 1.0
    return float(np.clip(np.mean(y) / pred_mean, lo, hi))


def shrunk_cohort_ratios(
    y: np.ndarray,
    pred: np.ndarray,
    cohort_weeks: np.ndarray,
    n_cohorts: int = config.N_COHORT_WEEKS,
    lo: float = 0.8,
    hi: float = 1.5,
) -> tuple[np.ndarray, dict]:
    """Per-cohort CIF recalibration factors with empirical-Bayes shrinkage.

    Raw per-cohort ratio r_c = realized lifetime default rate / mean predicted
    lifetime PD, computed on the calibration rows of cohort c. With ~200 loans
    per cohort the raw r_c is dominated by binomial noise (SE on the rate is
    ~0.028 at p~0.2), so fitting 13 free factors overfits the holdout. We shrink
    toward the global ratio r_g with the standard normal-normal EB estimator:

        f_c = r_g + tau^2 / (tau^2 + s_c^2) * (r_c - r_g)

    where s_c^2 = ybar_c(1-ybar_c) / (n_c * predbar_c^2) is the delta-method
    sampling variance of r_c (binomial numerator, denominator treated as fixed)
    and tau^2 is the method-of-moments between-cohort variance
    max(0, Var(r_c) - mean(s_c^2)). tau^2 = 0 (spread explained by sampling
    noise) degenerates gracefully to the global factor; large n_c earns more
    weight on the cohort's own ratio — no free hyperparameter. ybar_c is floored
    at 0.5/n_c in the variance so zero-default cohorts don't get s_c^2 = 0
    (which would wrongly disable shrinkage). Cohorts with no calibration rows
    get r_g. All factors are clipped to [lo, hi] like the global factor.

    Returns (factors, diag) with ``factors`` shape (n_cohorts,) indexed by
    cohort_week - 1, and ``diag`` carrying tau^2, r_g, and per-cohort raw
    ratios / sampling variances / shrink weights / sample sizes.
    """
    y = np.asarray(y, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    coh = np.asarray(cohort_weeks, dtype=float).ravel()

    pred_mean = float(np.mean(pred))
    r_g = float(np.mean(y) / pred_mean) if pred_mean > 1e-9 else 1.0

    r_raw = np.full(n_cohorts, np.nan)
    s2 = np.full(n_cohorts, np.nan)
    n_per = np.zeros(n_cohorts, dtype=int)
    for c in range(1, n_cohorts + 1):
        sel = coh == float(c)
        n = int(np.sum(sel))
        if n == 0:
            continue
        yb = float(np.mean(y[sel]))
        pb = float(np.mean(pred[sel]))
        if pb <= 1e-9:
            continue
        r_raw[c - 1] = yb / pb
        yb_f = float(np.clip(yb, 0.5 / n, 1.0 - 0.5 / n))
        s2[c - 1] = yb_f * (1.0 - yb_f) / (n * pb * pb)
        n_per[c - 1] = n

    obs = np.isfinite(r_raw)
    if int(np.sum(obs)) >= 2:
        tau2 = max(0.0, float(np.var(r_raw[obs], ddof=1)) - float(np.mean(s2[obs])))
    else:
        tau2 = 0.0

    factors = np.full(n_cohorts, r_g, dtype=float)
    weights = np.zeros(n_cohorts, dtype=float)
    if tau2 > 0.0:
        w = tau2 / (tau2 + s2[obs])
        factors[obs] = r_g + w * (r_raw[obs] - r_g)
        weights[obs] = w
    factors = np.clip(factors, lo, hi)

    diag = {
        "tau2": float(tau2),
        "global_ratio": float(np.clip(r_g, lo, hi)),
        "raw_ratios": [float(v) if np.isfinite(v) else None for v in r_raw],
        "sampling_var": [float(v) if np.isfinite(v) else None for v in s2],
        "shrink_weights": [float(v) for v in weights],
        "n_per_cohort": [int(v) for v in n_per],
        "factors": [float(v) for v in factors],
    }
    return factors, diag


def fit_cif_scales_per_cohort(
    models: list[HazardModel],
    calib_frame: pd.DataFrame,
    y_calib: np.ndarray,
    cohort_weeks: np.ndarray,
    lo: float = 0.8,
    hi: float = 1.5,
) -> tuple[np.ndarray, dict]:
    """Per-cohort CIF recalibration factors (EB-shrunk), fit out-of-time.

    Same calibration target as ``fit_cif_scale`` (realized lifetime default vs
    ensemble-mean predicted lifetime PD on a funded OOT set), but resolved per
    cohort_week with empirical-Bayes shrinkage toward the global factor — see
    ``shrunk_cohort_ratios`` for the estimator and the overfitting guard.
    """
    from . import features as _features

    y = np.asarray(y_calib, dtype=float)
    X = _features.build_features(calib_frame)
    preds = []
    for m in models:
        sc = [c for c in m.feature_cols if c != "loan_age_weeks"]
        preds.append(np.asarray(lifetime_pd(m, _features.align_columns(X, sc)), dtype=float))
    pred = np.mean(np.vstack(preds), axis=0)
    return shrunk_cohort_ratios(y, pred, cohort_weeks, lo=lo, hi=hi)


def per_loan_scale(
    cohort_weeks: np.ndarray,
    cohort_scales: np.ndarray,
    fallback: float = 1.0,
) -> np.ndarray:
    """Map per-cohort factors onto loans via their cohort_week (1-based).

    Loans with NaN / out-of-range cohort get ``fallback`` (the global factor).
    """
    coh = np.asarray(cohort_weeks, dtype=float).ravel()
    scales = np.asarray(cohort_scales, dtype=float).ravel()
    out = np.full(coh.shape[0], float(fallback), dtype=float)
    ok = np.isfinite(coh) & (coh >= 1) & (coh <= len(scales))
    out[ok] = scales[coh[ok].astype(int) - 1]
    return out


def predict_trajectory(
    models: list[HazardModel],
    decision_frame: pd.DataFrame,
    decisions: np.ndarray,
    cohort_weeks: np.ndarray,
    oot_iso=None,
    cif_scale: float = 1.0,
    cif_scale_by_cohort: np.ndarray | None = None,
) -> pd.DataFrame:
    """Produce the 169-row B grid (cohort_week x loan_age_weeks).

    For each ensemble member we compute CIF_d for approved applicants
    (decisions == 1), grouped by cohort_week, averaging within each cohort per
    weekly age. Across members we take the mean (point) and 5/95 percentiles
    (bounds), optionally apply the out-of-time isotonic calibrator ``oot_iso``
    (the same monotone map A uses, so cdr@age13 stays consistent with A's
    approved-set PD; monotonicity in age is preserved because isotonic is
    non-decreasing), enforce monotonicity in age per cohort, then clip_and_order.

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

    # Per-loan recalibration scale: per-cohort EB-shrunk factors when provided
    # (loans outside the cohort calendar fall back to the global cif_scale),
    # otherwise the single global factor.
    if cif_scale_by_cohort is not None:
        loan_scale = per_loan_scale(cohort_weeks, cif_scale_by_cohort, fallback=cif_scale)
    else:
        loan_scale = None

    # Per-model, per-cohort CIF curves: stack[m] has shape (n_cohorts, weeks).
    per_model_cohort = np.full((len(models), n_cohorts, weeks), np.nan, dtype=float)
    per_model_overall = np.full((len(models), weeks), np.nan, dtype=float)

    for m, model in enumerate(models):
        h_d, h_p = hazard_curves(model, X_full)
        cif = cif_default(h_d, h_p)  # (N, weeks)
        # WS2: out-of-time recalibration -- the hazard systematically under-predicts
        # on the OOT cohort (lifetime ratio ~1.12); a scale fit on the validation
        # funded subset corrects it (per-cohort EB-shrunk factors when supplied,
        # else one global factor). Clipped to [0,1].
        if loan_scale is not None:
            cif = np.clip(cif * loan_scale[:, None], 0.0, 1.0)
        elif cif_scale != 1.0:
            cif = np.clip(cif * cif_scale, 0.0, 1.0)

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

        # Out-of-time calibration (monotone) on the cumulative-incidence curve.
        if oot_iso is not None:
            point = calibration.apply_isotonic(oot_iso, point)
            lower = calibration.apply_isotonic(oot_iso, lower)
            upper = calibration.apply_isotonic(oot_iso, upper)

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

"""Approve/decline policy for Deliverable A -- maximize expected portfolio NPV.

The decision is NOT "approve if PD < threshold". It is "approve if the expected
NPV of funding this loan at its requested amount is positive at the conservative
(upper) PD bound", where the economics come from config (APR, term, origination
fee) and the recovery rate on default.

Economics (METHODOLOGY.md section 3):

    g = ORIGINATION_FEE_RATE + APR * TERM_DAYS / 365   (repaid-loan return rate)
    expected_npv(pi, amount, r) = amount * (g - pi * (g + 1 - r))

NPV is decreasing in pi, so requiring NPV > 0 at the UPPER PD bound makes
uncertain loans default to denial.
"""

from __future__ import annotations

import numpy as np

from . import config, economics


def _return_rate() -> float:
    """Repaid-loan return rate g from config constants."""
    return config.ORIGINATION_FEE_RATE + config.APR * config.TERM_DAYS / 365.0


def expected_npv(
    pi: np.ndarray,
    amount: np.ndarray,
    recovery_rate: float | np.ndarray = config.RECOVERY_PRIOR,
) -> np.ndarray:
    """Expected dollar NPV of funding each loan at `amount`.

        expected_npv = amount * (g - pi * (g + 1 - recovery_rate))

    where g = ORIGINATION_FEE_RATE + APR * TERM_DAYS / 365.
    """
    pi = np.asarray(pi, dtype=float)
    amount = np.asarray(amount, dtype=float)
    recovery_rate = np.asarray(recovery_rate, dtype=float)
    g = _return_rate()
    return amount * (g - pi * (g + 1.0 - recovery_rate))


def decide(
    pi_upper: np.ndarray,
    amount: np.ndarray,
    recovery_rate: float | np.ndarray = config.RECOVERY_PRIOR,
) -> np.ndarray:
    """Return 0/1 approve decisions: 1 where expected_npv(pi_upper, ...) > 0.

    `pi_upper` is the conservative upper 90% PD bound, so loans whose NPV is
    positive even at the upper bound are approved.
    """
    npv = expected_npv(pi_upper, amount, recovery_rate)
    return (npv > 0).astype(int)


def expected_profit(
    pd_hat: np.ndarray,
    amount: np.ndarray,
    recovery_rate: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Back-compat wrapper delegating to expected_npv."""
    return expected_npv(pd_hat, amount, recovery_rate)


# --------------------------------------------------------------------------- #
# WS1: portfolio decision rule (flat baseline vs timing-integrated)
# --------------------------------------------------------------------------- #


def conformal_adjusted_enpv(
    enpv_point: np.ndarray,
    pd_point: np.ndarray,
    p_def_mean: np.ndarray,
    amount: np.ndarray,
    half_width: float,
    recovery_interm: float = config.RECOVERY_PRIOR,
    recovery_spike: float = 0.0,
) -> np.ndarray:
    """E[NPV] re-priced at the conformal UPPER PD bound (WS3 folded into the decision).

    The split-conformal half-width (calibration.fit_pd_band) bounds how far the
    raw ensemble PD can sit below the empirical default rate. Consistent with the
    policy's "NPV at upper PD bound" design, the decision E[NPV] moves the
    half-width of probability mass from "repay" to "default", distributed over
    the loan's OWN predicted default-timing profile (so the timing rule's credit
    for late-in-term draws is preserved), and re-prices:

        enpv_conf_i = enpv_point_i + extra_i * R_i * (w_i - REPAID_NPV_RATE)

    where extra_i = min(hw, 1 - pd_i) and w_i is the per-unit default NPV
    averaged over loan i's default-week distribution. Loans with ~zero predicted
    default mass are charged at the worst case (day-90 total loss).
    """
    enpv_point = np.asarray(enpv_point, dtype=float).ravel()
    pd_point = np.asarray(pd_point, dtype=float).ravel()
    P = np.asarray(p_def_mean, dtype=float)
    amount = np.asarray(amount, dtype=float).ravel()
    hw = abs(float(half_width))

    npv_week_unit = economics._npv_default_unit_by_week(recovery_interm, recovery_spike)
    extra = np.clip(np.minimum(hw, 1.0 - pd_point), 0.0, 1.0)
    denom = np.maximum(pd_point, 1e-12)
    w = (P @ npv_week_unit) / denom
    w = np.where(pd_point < 1e-9, npv_week_unit[-1], w)
    return enpv_point + extra * amount * (w - economics.REPAID_NPV_RATE)


def oot_adjusted_enpv(
    enpv_point: np.ndarray,
    pd_point: np.ndarray,
    p_def_mean: np.ndarray,
    amount: np.ndarray,
    oot_factor: float,
    recovery_interm: float = config.RECOVERY_PRIOR,
    recovery_spike: float = 0.0,
) -> np.ndarray:
    """E[NPV] re-priced at the SIGNED out-of-time (OOT) lifetime-PD correction.

    The hazard model under-predicts default out-of-time by a measured signed
    factor (the B-recalibration lifetime ratio, ``survival.fit_cif_scale`` ~1.107).
    Deliverable B corrects its trajectory by this factor; this lever applies the
    same measured shift to A's lifetime PD before the E[NPV] decision. Scaling
    lifetime PD up by ``oot_factor`` moves ``(oot_factor - 1) * pd`` of probability
    mass from "repay" to "default", distributed over the loan's OWN predicted
    default-timing profile (so the timing rule's credit for late-in-term draws is
    preserved), and re-prices:

        enpv_oot_i = enpv_point_i + extra_i * R_i * (w_i - REPAID_NPV_RATE)

    where extra_i = clip((oot_factor - 1) * pd_i, 0, 1 - pd_i) and w_i is the
    per-unit default NPV averaged over loan i's default-week distribution. Loans
    with ~zero predicted default mass are charged at the worst case (day-90 total
    loss). Unlike the conformal lever this is a measured shift of the PD level,
    not a symmetric interval width.
    """
    enpv_point = np.asarray(enpv_point, dtype=float).ravel()
    pd_point = np.asarray(pd_point, dtype=float).ravel()
    P = np.asarray(p_def_mean, dtype=float)
    amount = np.asarray(amount, dtype=float).ravel()
    oot_factor = float(oot_factor)

    npv_week_unit = economics._npv_default_unit_by_week(recovery_interm, recovery_spike)
    extra = np.clip((oot_factor - 1.0) * pd_point, 0.0, 1.0 - pd_point)
    denom = np.maximum(pd_point, 1e-12)
    w = (P @ npv_week_unit) / denom
    w = np.where(pd_point < 1e-9, npv_week_unit[-1], w)
    return enpv_point + extra * amount * (w - economics.REPAID_NPV_RATE)


def portfolio_decisions(
    models,
    X_static,
    amount,
    recovery_interm: float = config.RECOVERY_PRIOR,
    recovery_spike: float = 0.0,
    *,
    rule: str = "timing",
    conservative: bool = True,
    conformal_hw: float | None = None,
    lo_pct: float = config.INTERVAL_LOWER_Q * 100.0,
    hi_pct: float = config.INTERVAL_UPPER_Q * 100.0,
) -> dict:
    """Score a frame with the hazard ensemble and decide approve/decline.

    Two rules share one interface so the scorecard can ablate them and the
    pipeline can switch via ``config.POLICY_RULE``:

      ``"flat"``    -- the break-even baseline: expected_npv linear in lifetime
                       PD, decided at the conservative (upper) PD bound. This
                       reproduces the pre-WS1 behaviour.
      ``"timing"``  -- WS1: E[NPV] integrated over the per-week default-timing
                       distribution (economics.expected_npv_timing), giving
                       credit for the daily draws collected before a *late*
                       in-term default. Decided at the conservative lower E[NPV]
                       bound across ensemble members.

    Approve iff the (conservative) expected NPV is positive.

    ``conformal_hw`` (WS3->decision lever, config.POLICY_CONFORMAL_DECISION):
    when given, the decision basis is instead the E[NPV] evaluated at the
    conformal upper PD bound ``pd_point + conformal_hw`` (see
    conformal_adjusted_enpv). The half-width must be fit on data DISJOINT from
    what the decision is evaluated on. Reported PD columns are unchanged.

    Returns a dict of float arrays (all shape (N,)):
        decision (int 0/1), predicted_pd, pd_lower_90, pd_upper_90,
        enpv_point, enpv_lower, enpv_upper
        (+ enpv_conformal when conformal_hw is given).
    """
    from . import calibration, features, survival  # local: avoid import cycles

    amount = np.asarray(amount, dtype=float)
    model_list = list(models) if isinstance(models, (list, tuple)) else [models]

    pd_rows: list[np.ndarray] = []
    enpv_rows: list[np.ndarray] = []
    p_def_rows: list[np.ndarray] = []
    for m in model_list:
        static_cols = [c for c in m.feature_cols if c != "loan_age_weeks"]
        Xa = features.align_columns(X_static, static_cols)
        if rule == "timing":
            p_def, p_pay = survival.default_week_probs(m, Xa)
            pd_vec = p_def.sum(axis=1)
            enpv = economics.expected_npv_timing(
                p_def, p_pay, amount, recovery_interm, recovery_spike
            )
            p_def_rows.append(p_def)
        elif rule == "flat":
            pd_vec = np.asarray(survival.lifetime_pd(m, Xa), dtype=float).ravel()
            enpv = expected_npv(pd_vec, amount, recovery_interm)
        else:
            raise ValueError(f"rule must be 'flat' or 'timing', got {rule!r}")
        pd_rows.append(pd_vec)
        enpv_rows.append(np.asarray(enpv, dtype=float).ravel())

    pd_samples = np.vstack(pd_rows)
    enpv_samples = np.vstack(enpv_rows)

    pd_lower, pd_point, pd_upper = calibration.ensemble_intervals(
        pd_samples, lo=lo_pct / 100.0, hi=hi_pct / 100.0
    )
    enpv_point = np.mean(enpv_samples, axis=0)
    enpv_lower = np.percentile(enpv_samples, lo_pct, axis=0)
    enpv_upper = np.percentile(enpv_samples, hi_pct, axis=0)

    out_extra: dict = {}
    if conformal_hw is not None:
        if rule == "timing":
            p_def_mean = np.mean(np.stack(p_def_rows), axis=0)
            enpv_conf = conformal_adjusted_enpv(
                enpv_point, pd_point, p_def_mean, amount,
                float(conformal_hw), recovery_interm, recovery_spike,
            )
        else:
            pd_up = np.clip(pd_point + abs(float(conformal_hw)), 0.0, 1.0)
            enpv_conf = expected_npv(pd_up, amount, recovery_interm)
        basis = enpv_conf
        out_extra["enpv_conformal"] = enpv_conf
    else:
        basis = enpv_lower if conservative else enpv_point
    decision = (basis > 0).astype(int)
    decision = np.where(np.isfinite(amount), decision, 0).astype(int)

    return {
        "decision": decision,
        "predicted_pd": pd_point,
        "pd_lower_90": pd_lower,
        "pd_upper_90": pd_upper,
        "enpv_point": enpv_point,
        "enpv_lower": enpv_lower,
        "enpv_upper": enpv_upper,
        **out_extra,
    }

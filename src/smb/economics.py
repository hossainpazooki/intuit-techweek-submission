"""Loan NPV economics -- the single source of truth for dollar outcomes.

Two distinct computations live here, both derived from the challenge brief (p.8):

1. ``realized_npv`` -- the *backtest* NPV of a matured loan from its REALIZED
   outcome (default flag, day-of-default, recovered amount). Used by the P&L
   proxy in scripts/run_scorecard.py to score a policy against ground truth.

2. ``expected_npv_timing`` -- the *forward* expected NPV of funding a loan,
   integrated over the per-week default-timing distribution from the
   competing-risks hazard (survival.default_week_probs). Used by policy.py to
   decide approve/decline. This is the WS1 upgrade over the flat break-even rule.

Brief economics (per unit principal R, i.e. multiply by ``amount`` for dollars):

    F      = 0.03                       origination fee, collected up front
    iT     = APR * TERM_DAYS / 365      total interest over the 60-day term
           = 0.35 * 60 / 365 = 0.0575
    repaid net NPV          = F + iT                = 0.0875            (per unit)
    daily draw D (per unit) = (1 + iT) / TERM_DAYS  = 1.0575 / 60
    default at day t* net   = F + D*(t*-1) + rec - 1                    (per unit)

KEY CONVENTION (recomputed from raw CSVs, documented in METHODOLOGY/LEARNINGS):
default timing in the data is bimodal -- in-term defaults span days 3..60, then a
spike at *exactly* day 90 (the observation-window close) with **zero** mass in
(60, 90). The day-90 spike loans recover ~0 on average (empirical mean recovery
fraction 0.0001 vs 0.118 for in-term defaults), i.e. they are near-total losses
that never amortized. Cash flow is bounded by the 60-day contractual term, so we
credit daily draws ONLY for in-term defaults (t* <= 60); day-90-window defaults
get no term draws (their realized cash is the fee plus whatever was recovered):

    in-term default (t* <= 60):   NPV = F + D*(t*-1) + rec - 1
    day-90 default  (t* > 60):    NPV = F          + rec - 1
"""

from __future__ import annotations

import numpy as np

from . import config

# --------------------------------------------------------------------------- #
# Per-unit-principal constants (brief p.8)
# --------------------------------------------------------------------------- #

F_RATE = config.ORIGINATION_FEE_RATE                       # 0.03
INTEREST_TERM_RATE = config.APR * config.TERM_DAYS / 365.0  # 0.0575
REPAID_NPV_RATE = F_RATE + INTEREST_TERM_RATE              # 0.0875 (per unit)
DAILY_DRAW_UNIT = (1.0 + INTEREST_TERM_RATE) / config.TERM_DAYS  # 1.0575/60 per unit/day

# Weeks 1..IN_TERM_LAST_WEEK are in-term (defaults amortize, draws credited);
# later weeks are the day-90 observation-window region (total-loss convention).
# terminal_week(60) == 9 with the survival_data weekly mapping.
IN_TERM_LAST_WEEK = 9


# --------------------------------------------------------------------------- #
# Realized (backtest) NPV
# --------------------------------------------------------------------------- #


def realized_npv(
    default_flag: np.ndarray,
    days_to_default: np.ndarray,
    recovered_amount: np.ndarray,
    amount: np.ndarray,
) -> np.ndarray:
    """Realized dollar NPV per loan from its matured outcome (vectorized).

    Args:
        default_flag: 1.0 if the loan defaulted, 0.0 if repaid (per loan).
        days_to_default: realized day-of-default t* (ignored for repaid loans).
        recovered_amount: absolute dollars recovered post-default (NaN -> 0).
        amount: loan principal R in dollars.

    Returns:
        Array of realized NPV in dollars, one per loan.
    """
    flag = np.asarray(default_flag, dtype=float)
    t = np.asarray(days_to_default, dtype=float)
    rec = np.nan_to_num(np.asarray(recovered_amount, dtype=float), nan=0.0)
    R = np.asarray(amount, dtype=float)

    repaid = REPAID_NPV_RATE * R

    in_term = flag == 1.0
    is_spike = in_term & (t > config.TERM_DAYS)
    is_interm = in_term & (t <= config.TERM_DAYS)

    draws = np.clip(t - 1.0, 0.0, config.TERM_DAYS)  # daily draws collected
    npv_interm = F_RATE * R + DAILY_DRAW_UNIT * draws * R + rec - R
    npv_spike = F_RATE * R + rec - R

    out = np.where(in_term, np.where(is_spike, npv_spike, npv_interm), repaid)
    return out.astype(float)


# --------------------------------------------------------------------------- #
# Forward expected NPV, integrated over default timing (WS1)
# --------------------------------------------------------------------------- #


def _npv_default_unit_by_week(
    recovery_interm: float,
    recovery_spike: float,
) -> np.ndarray:
    """Per-unit NPV of a default occurring in each weekly age 1..WEEKS.

    In-term weeks credit the daily draws collected up to a representative day
    (capped at the 60-day term); day-90-window weeks are the total-loss case.
    """
    weeks = config.WEEKS
    out = np.empty(weeks, dtype=float)
    for t in range(1, weeks + 1):
        if t <= IN_TERM_LAST_WEEK:
            rep_day = min(7 * t, config.TERM_DAYS)  # representative day-of-default
            draws = rep_day - 1
            out[t - 1] = F_RATE + DAILY_DRAW_UNIT * draws + recovery_interm - 1.0
        else:
            out[t - 1] = F_RATE + recovery_spike - 1.0
    return out


def expected_npv_timing(
    default_week_probs: np.ndarray,
    payoff_prob: np.ndarray,
    amount: np.ndarray,
    recovery_interm: float = config.RECOVERY_PRIOR,
    recovery_spike: float = 0.0,
) -> np.ndarray:
    """Forward expected dollar NPV, integrated over default-timing.

        E[NPV_i] = sum_t P(default in week t) * NPV_default(t)
                   + P(repay) * 0.0875        (all per unit, * amount)

    Args:
        default_week_probs: shape (N, WEEKS); P(default in weekly age t) per loan
            (from survival.default_week_probs).
        payoff_prob: shape (N,); P(loan repays / pays off) per loan.
        amount: principal R in dollars, shape (N,).
        recovery_interm: expected recovery fraction for in-term defaults.
        recovery_spike: expected recovery fraction for day-90-window defaults.

    Returns:
        Expected NPV in dollars, shape (N,).
    """
    P = np.asarray(default_week_probs, dtype=float)
    if P.ndim == 1:
        P = P[np.newaxis, :]
    payoff = np.asarray(payoff_prob, dtype=float).ravel()
    R = np.asarray(amount, dtype=float).ravel()

    npv_week_unit = _npv_default_unit_by_week(recovery_interm, recovery_spike)  # (WEEKS,)
    enpv_unit = P @ npv_week_unit + payoff * REPAID_NPV_RATE
    return enpv_unit * R

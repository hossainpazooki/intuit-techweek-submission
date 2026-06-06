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

from . import config


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

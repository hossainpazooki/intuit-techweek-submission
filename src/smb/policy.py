"""Approve/decline policy for Deliverable A — maximize expected portfolio profit.

The decision is NOT "approve if PD < threshold". It's "approve if the expected
profit of funding this loan at its requested amount is positive", where profit
depends on the loan economics in config (APR, term, origination fee) and the
recovery on default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def expected_profit(
    pd_hat: np.ndarray,
    amount: np.ndarray,
    recovery_rate: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Expected dollar profit of funding each loan at `amount`.

    Sketch of the economics (refine against the brief):

        revenue_if_paid = origination_fee + interest_earned(amount, APR, TERM_DAYS)
        loss_if_default = amount * (1 - recovery_rate) - partial_revenue_collected

        E[profit] = (1 - pd_hat) * revenue_if_paid - pd_hat * loss_if_default

    TODO: model the daily-ACH amortization so interest/partial collection on
    defaulted loans isn't ignored; estimate recovery_rate from
    final_recovered_amount on labeled defaults.
    """
    raise NotImplementedError


def decide(pd_hat: np.ndarray, amount: np.ndarray, **kwargs) -> np.ndarray:
    """Return 0/1 approve decisions. approve where expected_profit > 0.

    Optionally apply a profit margin / risk-appetite buffer instead of a hard
    zero threshold, and document the tradeoff in Deliverable D.
    """
    raise NotImplementedError

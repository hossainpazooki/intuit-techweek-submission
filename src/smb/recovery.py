"""Recovery-rate estimation for defaulted loans.

The NPV economics (policy.expected_npv) need a recovery rate: the fraction of a
defaulted loan's principal that is eventually recovered. We estimate it
empirically from labeled, matured defaulters as

    recovery = mean( final_recovered_amount / requested_amount )

over rows that actually defaulted, falling back to the configured prior
(config.RECOVERY_PRIOR) when no usable defaulters are available. The result is
clamped to a sane [0, 0.62] band (the dataset's recovered amounts never exceed
~62% of principal).

The baseline pipeline only needs this scalar, so we keep it deliberately simple.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smb import config

# Empirical upper bound on recovery as a fraction of principal.
_RECOVERY_MAX = 0.62


def estimate_recovery_rate(labeled_train: pd.DataFrame) -> float:
    """Mean recovered fraction over defaulters, clamped to [0, 0.62].

    Parameters
    ----------
    labeled_train
        Frame of labeled (funded + matured) loans. Expected columns:
        ``default_flag``, ``final_recovered_amount``, ``requested_amount``.

    Returns
    -------
    float
        Estimated recovery rate in [0, 0.62]; ``config.RECOVERY_PRIOR`` when no
        usable defaulters are present.
    """
    fallback = float(np.clip(config.RECOVERY_PRIOR, 0.0, _RECOVERY_MAX))

    required = {"default_flag", "final_recovered_amount", "requested_amount"}
    if labeled_train is None or not required.issubset(labeled_train.columns):
        return fallback

    flag = pd.to_numeric(labeled_train["default_flag"], errors="coerce")
    recovered = pd.to_numeric(labeled_train["final_recovered_amount"], errors="coerce")
    principal = pd.to_numeric(labeled_train["requested_amount"], errors="coerce")

    # Defaulters with a valid, positive principal and a usable recovery value.
    mask = (
        (flag == 1)
        & recovered.notna()
        & principal.notna()
        & (principal > 0)
    )
    if not bool(mask.any()):
        return fallback

    ratio = (recovered[mask] / principal[mask]).to_numpy(dtype=float)
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size == 0:
        return fallback

    rate = float(np.mean(ratio))
    if not np.isfinite(rate):
        return fallback

    return float(np.clip(rate, 0.0, _RECOVERY_MAX))


def estimate_recovery_rates_by_timing(
    labeled_train: pd.DataFrame,
) -> tuple[float, float]:
    """Mean recovered fraction split by default timing: (in-term, day-90 spike).

    In-term defaulters (days_to_default <= 60) amortize and recover materially;
    day-90-window defaulters (> 60) recover ~0 empirically. The timing-integrated
    E[NPV] (economics.expected_npv_timing) prices these two regimes separately,
    so we return both rates. Each is clamped to [0, 0.62]; missing data falls
    back to the overall scalar / prior.
    """
    overall = estimate_recovery_rate(labeled_train)
    required = {"default_flag", "final_recovered_amount", "requested_amount", "days_to_default"}
    if labeled_train is None or not required.issubset(labeled_train.columns):
        return overall, 0.0

    flag = pd.to_numeric(labeled_train["default_flag"], errors="coerce")
    recovered = pd.to_numeric(labeled_train["final_recovered_amount"], errors="coerce")
    principal = pd.to_numeric(labeled_train["requested_amount"], errors="coerce")
    days = pd.to_numeric(labeled_train["days_to_default"], errors="coerce")

    base = (flag == 1) & recovered.notna() & principal.notna() & (principal > 0)
    ratio = (recovered / principal)

    def _mean(mask: pd.Series, fallback: float) -> float:
        vals = ratio[mask].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return fallback
        return float(np.clip(np.mean(vals), 0.0, _RECOVERY_MAX))

    interm = _mean(base & (days <= config.TERM_DAYS), overall)
    spike = _mean(base & (days > config.TERM_DAYS), 0.0)
    return interm, spike

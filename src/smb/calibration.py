"""Shared 90% prediction-interval machinery for A, B, and C.

The scorer grades interval *calibration* (do ~90% of true values land inside?)
against interval *width*. Calibrate on held-out data with known outcomes —
validation.csv has outcomes, so it's the natural calibration set.
"""

from __future__ import annotations

import numpy as np

from . import config


def conformal_intervals(
    point: np.ndarray,
    residual_quantiles: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Build [lower, upper] around point estimates via (split) conformal offsets.

    TODO: derive residual_quantiles from calibration-set errors so empirical
    coverage ~= 90%. Clip to [0, 1] and guarantee lower <= point <= upper before
    returning (the validator rejects any violation).
    """
    raise NotImplementedError


def clip_and_order(
    lower: np.ndarray, point: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Final safety net: clip to [0,1] and enforce lower <= point <= upper."""
    lower = np.clip(lower, 0.0, 1.0)
    point = np.clip(point, 0.0, 1.0)
    upper = np.clip(upper, 0.0, 1.0)
    lower = np.minimum(lower, point)
    upper = np.maximum(upper, point)
    return lower, point, upper


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Empirical fraction of truths inside [lower, upper] — target ~0.90."""
    return float(np.mean((y_true >= lower) & (y_true <= upper)))

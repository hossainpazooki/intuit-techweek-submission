"""Shared 90% prediction-interval machinery for A, B, and C.

The scorer grades interval *calibration* (do ~90% of true values land inside?)
against interval *width*. Calibrate on held-out data with known outcomes —
validation.csv has outcomes, so it's the natural calibration set.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

from . import config


def conformal_intervals(
    point: np.ndarray,
    residual_quantiles: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Build [lower, upper] around point estimates via (split) conformal offsets.

    ``residual_quantiles`` is ``(lower_offset, upper_offset)`` derived from
    calibration-set errors. The lower bound is ``point - lower_offset`` and the
    upper bound is ``point + upper_offset``. The result is clipped to [0, 1] and
    forced to satisfy ``lower <= point <= upper`` via ``clip_and_order``.
    """
    point = np.asarray(point, dtype=float)
    lower_offset, upper_offset = residual_quantiles
    lower = point - abs(float(lower_offset))
    upper = point + abs(float(upper_offset))
    lower, point, upper = clip_and_order(lower, point, upper)
    return lower, upper


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


def fit_isotonic(p: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    """Fit a monotone probability calibrator mapping raw scores p -> outcomes y."""
    p = np.asarray(p, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p, y)
    return iso


def apply_isotonic(iso: IsotonicRegression, p: np.ndarray) -> np.ndarray:
    """Apply a fitted isotonic calibrator and clip to [0, 1]."""
    p = np.asarray(p, dtype=float).ravel()
    out = iso.predict(p)
    return np.clip(out, 0.0, 1.0)


def reliability_and_ece(
    p: np.ndarray, y: np.ndarray, n_bins: int = 10
) -> dict:
    """Expected calibration error (ECE) plus per-bin reliability stats.

    Returns a dict with ``ece`` (float) and ``bins`` (list of dicts describing
    each bin: edges, count, mean predicted probability, observed frequency).
    """
    p = np.asarray(p, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = p.shape[0]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict] = []
    ece = 0.0
    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        count = int(np.sum(mask))
        if count > 0:
            mean_pred = float(np.mean(p[mask]))
            obs_freq = float(np.mean(y[mask]))
            ece += (count / n) * abs(mean_pred - obs_freq)
        else:
            mean_pred = float("nan")
            obs_freq = float("nan")
        bins.append(
            {
                "lower": float(lo),
                "upper": float(hi),
                "count": count,
                "mean_pred": mean_pred,
                "obs_freq": obs_freq,
            }
        )
    return {"ece": float(ece), "bins": bins}


def fit_pd_band(
    point_cal: np.ndarray,
    y_cal: np.ndarray,
    band_quantile: float = 0.95,
    n_bins: int = 10,
) -> float:
    """Split-conformal half-width for a 90% PD band, from a set with outcomes.

    The raw ensemble percentile bands measure model *disagreement*, not predictive
    uncertainty for the binary default outcome, so they under-cover. We instead
    bin loans by predicted PD and take the ``band_quantile`` quantile of per-bin
    |empirical_rate - mean_predicted| as a symmetric half-width.

    Crucially the conformity is measured on the RAW ensemble point -- NOT a
    recalibrated point. A fitted recalibrator (isotonic/global-shift) shrinks the
    in-fold reliability error, so its half-width under-covers out-of-fold; the raw
    half-width generalizes (verified: raw->0.875 vs isotonic->0.53 OOF coverage at
    the same target). ``band_quantile=0.95`` (slightly above 0.90) absorbs the
    finite-sample binomial slack to land binned coverage near 0.90.
    """
    point_cal = np.asarray(point_cal, dtype=float).ravel()
    y_cal = np.asarray(y_cal, dtype=float).ravel()
    order = np.argsort(point_cal)
    conf = [
        abs(float(np.mean(y_cal[b])) - float(np.mean(point_cal[b])))
        for b in np.array_split(order, n_bins)
        if len(b) > 0
    ]
    return float(np.quantile(conf, band_quantile)) if conf else 0.0


def apply_pd_band(
    half_width: float, point_apply: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a conformal half-width -> (lower, point, upper) around the point."""
    p = np.asarray(point_apply, dtype=float).ravel()
    return clip_and_order(p - half_width, p, p + half_width)


def ensemble_intervals(
    samples: np.ndarray,
    lo: float = 0.05,
    hi: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Summarize ensemble draws into (lower, point, upper).

    ``samples`` has shape (n_models, n). The point estimate is the mean across
    models; the bounds are the ``lo``/``hi`` percentiles across models. The
    result is clipped and ordered.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        samples = samples[np.newaxis, :]
    point = np.mean(samples, axis=0)
    lower = np.percentile(samples, lo * 100.0, axis=0)
    upper = np.percentile(samples, hi * 100.0, axis=0)
    lower, point, upper = clip_and_order(lower, point, upper)
    return lower, point, upper

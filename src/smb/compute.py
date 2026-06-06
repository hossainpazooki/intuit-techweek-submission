"""Compute-budget knobs: --compute {low,med,high} -> a frozen dict of settings.

Everything that trades runtime for quality reads its sizing from here so a single
flag scales the whole pipeline AND stays deterministic per (seed, budget):

  bag_size        : PD/hazard ensemble members (more -> tighter, smoother PD)
  hp_width        : hyperparameter-search width multiplier (1 = baseline grid)
  conformal_folds : split-conformal CV folds for A/B intervals
  boot_b          : bootstrap replicates for Deliverable B trajectory bands
  boot_c          : bootstrap replicates for Deliverable C counterfactual bands
  enpv_lo_pct     : lower percentile used for the conservative E[NPV] decision
  sev_resolution  : harness severity-grid points (declined-coverage curve)
  sev_seeds       : harness seeds per severity point

The numbers grow monotonically low -> med -> high; the curve in
artifacts/compute_curve.csv should show weighted_proxy rising (or plateauing)
with budget.
"""

from __future__ import annotations

from types import MappingProxyType

LEVELS = ("low", "med", "high")

_BUDGETS = {
    "low": {
        "bag_size": 3,
        "hp_width": 1,
        "conformal_folds": 3,
        "boot_b": 50,
        "boot_c": 50,
        "enpv_lo_pct": 5.0,
        "sev_resolution": 3,
        "sev_seeds": 2,
    },
    "med": {
        "bag_size": 5,
        "hp_width": 2,
        "conformal_folds": 5,
        "boot_b": 120,
        "boot_c": 120,
        "enpv_lo_pct": 5.0,
        "sev_resolution": 5,
        "sev_seeds": 4,
    },
    "high": {
        "bag_size": 8,
        "hp_width": 3,
        "conformal_folds": 10,
        "boot_b": 250,
        "boot_c": 250,
        "enpv_lo_pct": 5.0,
        "sev_resolution": 9,
        "sev_seeds": 8,
    },
}


def budget(level: str = "med") -> MappingProxyType:
    """Return the read-only settings dict for a compute level."""
    key = str(level).strip().lower()
    if key not in _BUDGETS:
        raise ValueError(f"compute level must be one of {LEVELS}, got {level!r}")
    return MappingProxyType(dict(_BUDGETS[key]))

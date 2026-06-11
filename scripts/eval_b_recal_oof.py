"""Out-of-fold adopt/reject evidence for per-cohort OOT recalibration (B/WS2).

The WS2 recalibration (global lifetime ratio, and the per-cohort EB-shrunk
factors that generalize it) is fit on the validation funded holdout -- the same
2,551 loans the trajectory proxy is scored on. Any in-fold gain from 13 free
parameters is therefore INADMISSIBLE evidence. This script measures both arms
strictly out-of-fold:

  repeated cohort-stratified 50/50 splits of the holdout; per split, BOTH the
  global ratio and the per-cohort EB-shrunk factors are fit on one half only,
  and the cohort-size-weighted CDR MAE (same metric as run_scorecard.traj_proxy)
  is measured on the disjoint other half. Both directions of each split are
  used; the comparison is paired (same split, same CIF predictions, same
  realized CDRs -- only the recalibration differs).

Why not leave-one-cohort-out: under LOCO the evaluated cohort contributes no
calibration rows, so its EB factor degenerates to the global ratio by
construction and the two arms are identical -- LOCO cannot distinguish them.
The repeated-split design keeps each cohort's *other* rows available for its
factor while never letting an evaluated row into the fit, which is exactly the
deployment situation (factors fit on validation funded, applied to unseen
loans of the same cohorts).

The hazard ensemble itself is fit on TRAIN only (out-of-time wrt the holdout),
so its CIF predictions are computed once and shared by both arms and all folds.

Usage:
    python scripts/eval_b_recal_oof.py [--compute {low,med,high}] [--splits N]

Writes reports/b_recal_oof.json. Deterministic per (seed, compute, splits).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smb import (  # noqa: E402
    compute as compute_mod,
    config,
    data,
    features,
    survival,
)

CLIP_LO, CLIP_HI = 0.8, 1.5  # same clipping as survival.fit_cif_scale


def weighted_cdr_mae(
    cif: np.ndarray,
    flag: np.ndarray,
    days: np.ndarray,
    cohort: np.ndarray,
    idx: np.ndarray,
    loan_scale: np.ndarray,
) -> float:
    """Cohort-size-weighted MAE of predicted vs realized CDR_{w,a} on rows idx.

    Identical metric to run_scorecard.traj_proxy: realized CDR for cohort w at
    weekly age a = fraction of cohort-w rows defaulted by day 7a; predicted =
    mean scaled CIF over the same rows; weight = cohort size.
    """
    weeks = config.WEEKS
    cif_s = np.clip(cif[idx] * loan_scale[idx, None], 0.0, 1.0)
    coh = cohort[idx]
    fl = flag[idx]
    dy = days[idx]

    abs_err_w = 0.0
    n_w = 0.0
    for c in np.unique(coh[np.isfinite(coh)]):
        sel = coh == c
        ncoh = int(np.sum(sel))
        if ncoh == 0:
            continue
        for a in range(weeks):
            realized = float(np.mean((fl[sel] == 1) & (dy[sel] <= 7 * (a + 1))))
            pred = float(np.mean(cif_s[sel, a]))
            abs_err_w += ncoh * abs(pred - realized)
            n_w += ncoh
    return abs_err_w / n_w if n_w else float("nan")


def stratified_halves(cohort: np.ndarray, rng: np.random.Generator):
    """Cohort-stratified 50/50 split -> (idx_a, idx_b)."""
    a_parts, b_parts = [], []
    for c in np.unique(cohort[np.isfinite(cohort)]):
        idx = np.flatnonzero(cohort == c)
        idx = rng.permutation(idx)
        half = len(idx) // 2
        a_parts.append(idx[:half])
        b_parts.append(idx[half:])
    # Rows with NaN cohort (none expected on the holdout) go to side b.
    nan_idx = np.flatnonzero(~np.isfinite(cohort))
    if len(nan_idx):
        b_parts.append(nan_idx)
    return np.concatenate(a_parts), np.concatenate(b_parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compute", choices=compute_mod.LEVELS, default="med")
    ap.add_argument("--splits", type=int, default=25)
    args = ap.parse_args(argv)

    budget = compute_mod.budget(args.compute)
    ds = data.load_dataset()
    labeled = ds.labeled_train
    holdout = ds.validation[ds.validation["default_flag"].notna()].reset_index(drop=True)

    seeds = [config.RANDOM_SEED + i for i in range(budget["bag_size"])]
    print(f"[1/3] fitting hazard ensemble on TRAIN (compute={args.compute}, "
          f"{len(seeds)} seeds)...")
    models = survival.fit_hazard_ensemble(labeled, seeds)

    print("[2/3] scoring holdout CIF once (shared by both arms / all folds)...")
    X = features.build_features(holdout)
    cif_stack = []
    for m in models:
        static_cols = [c for c in m.feature_cols if c != "loan_age_weeks"]
        Xa = features.align_columns(X, static_cols)
        h_d, h_p = survival.hazard_curves(m, Xa)
        cif_stack.append(survival.cif_default(h_d, h_p))
    cif = np.mean(cif_stack, axis=0)  # (N, WEEKS), unscaled
    pred_life = cif[:, -1]

    flag = holdout["default_flag"].to_numpy(float)
    days = pd.to_numeric(holdout["days_to_default"], errors="coerce").to_numpy(float)
    cohort = data.assign_cohort_week(holdout).to_numpy(float)
    n = len(holdout)

    print(f"[3/3] repeated cohort-stratified 50/50 splits (x{args.splits}, "
          f"both directions)...")
    rows = []
    for s in range(args.splits):
        rng = np.random.default_rng(config.RANDOM_SEED + s)
        a, b = stratified_halves(cohort, rng)
        for fit_idx, test_idx in ((a, b), (b, a)):
            y_fit = flag[fit_idx]
            p_fit = pred_life[fit_idx]
            pm = float(np.mean(p_fit))
            r_g = float(np.clip(np.mean(y_fit) / pm, CLIP_LO, CLIP_HI)) if pm > 1e-9 else 1.0

            # Arm 1: global single factor (baseline).
            scale_g = np.full(n, r_g)
            mae_g = weighted_cdr_mae(cif, flag, days, cohort, test_idx, scale_g)

            # Arm 2: per-cohort EB-shrunk factors (candidate).
            factors, diag = survival.shrunk_cohort_ratios(
                y_fit, p_fit, cohort[fit_idx], lo=CLIP_LO, hi=CLIP_HI
            )
            scale_c = survival.per_loan_scale(cohort, factors, fallback=r_g)
            mae_c = weighted_cdr_mae(cif, flag, days, cohort, test_idx, scale_c)

            rows.append({
                "split": s, "mae_global": mae_g, "mae_per_cohort": mae_c,
                "diff": mae_c - mae_g, "tau2": diag["tau2"], "r_g": r_g,
            })

    mg = np.array([r["mae_global"] for r in rows])
    mc = np.array([r["mae_per_cohort"] for r in rows])
    diff = mc - mg
    wins = int(np.sum(diff < 0))

    # Full-holdout fit (the factors that actually ship) -- reference only,
    # in-fold by construction, NOT adopt evidence.
    factors_full, diag_full = survival.shrunk_cohort_ratios(
        flag, pred_life, cohort, lo=CLIP_LO, hi=CLIP_HI
    )
    pm_full = float(np.mean(pred_life))
    r_g_full = float(np.clip(np.mean(flag) / pm_full, CLIP_LO, CLIP_HI))
    all_idx = np.arange(n)
    infold_g = weighted_cdr_mae(cif, flag, days, cohort, all_idx, np.full(n, r_g_full))
    infold_c = weighted_cdr_mae(
        cif, flag, days, cohort, all_idx,
        survival.per_loan_scale(cohort, factors_full, fallback=r_g_full),
    )

    out = {
        "_design": "repeated cohort-stratified 50/50 splits of the validation "
        "funded holdout; recalibration (both arms) fit on one half, weighted "
        "CDR MAE measured on the disjoint half; paired; hazard ensemble fit on "
        "TRAIN only.",
        "compute": args.compute,
        "seed": config.RANDOM_SEED,
        "n_splits": args.splits,
        "n_evals": len(rows),
        "n_holdout": n,
        "oof_mae_global_mean": float(np.mean(mg)),
        "oof_mae_per_cohort_mean": float(np.mean(mc)),
        "oof_diff_mean": float(np.mean(diff)),
        "oof_diff_std": float(np.std(diff, ddof=1)),
        "oof_diff_se": float(np.std(diff, ddof=1) / np.sqrt(len(diff))),
        "per_cohort_wins": wins,
        "per_cohort_losses": int(np.sum(diff > 0)),
        "ties": int(np.sum(diff == 0)),
        "infold_reference": {
            "mae_global": float(infold_g),
            "mae_per_cohort": float(infold_c),
            "note": "fit and scored on the same rows -- NOT adopt evidence",
        },
        "full_fit_diag": diag_full,
        "folds": rows,
    }
    dest = ROOT / "reports" / "b_recal_oof.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 70)
    print("OUT-OF-FOLD weighted CDR MAE (lower is better)")
    print(f"  global (baseline)     mean = {np.mean(mg):.5f}")
    print(f"  per-cohort EB (cand)  mean = {np.mean(mc):.5f}")
    print(f"  paired diff (cand-base) = {np.mean(diff):+.5f} "
          f"(SE {np.std(diff, ddof=1) / np.sqrt(len(diff)):.5f}); "
          f"cand wins {wins}/{len(rows)}")
    print(f"  in-fold reference: global {infold_g:.5f}  per-cohort {infold_c:.5f}")
    print(f"  full-fit tau^2 = {diag_full['tau2']:.5f}  "
          f"factors = {[round(f, 3) for f in diag_full['factors']]}")
    print("=" * 70)
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

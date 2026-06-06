"""Proxy scorecard for the SMB Underwriting Challenge.

CAVEAT (printed at the top of every run): the TRUE official score is not locally
computable -- the test labels and the per-term normalization are hidden. This
scorecard optimizes against PROXIES measured only where ground truth exists
(train funded+matured loans for the fit; the out-of-time *validation funded*
subset, 2,551 loans applied after the train window, as the held-out evaluation),
aggregated with the brief's published weights:

    S = 0.30*S_P&L + 0.25*S_traj + 0.20*S_cal + 0.10*S_C + 0.15*S_write

Every empirical number is recomputed from the raw CSVs. All randomness flows from
config.RANDOM_SEED so the scorecard is byte-identical per (seed, compute budget).

Usage:
    python scripts/run_scorecard.py --compute {low,med,high} [--out PATH]
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
    calibration,
    compute as compute_mod,
    config,
    data,
    economics,
    features,
    policy,
    recovery,
    survival,
)

HARNESS_SRC = ROOT.parent / "closed-loop-default-detection" / "src"


# --------------------------------------------------------------------------- #
# Holdout construction
# --------------------------------------------------------------------------- #


def _load_holdout(ds: data.Dataset):
    """Out-of-time validation funded subset (has realized outcomes)."""
    val = ds.validation
    holdout = val[val["default_flag"].notna()].reset_index(drop=True)
    return holdout


def _fit_models(labeled: pd.DataFrame, bag_size: int):
    seeds = [config.RANDOM_SEED + i for i in range(bag_size)]
    return survival.fit_hazard_ensemble(labeled, seeds)


# --------------------------------------------------------------------------- #
# P&L proxy (S_P&L, 30%)
# --------------------------------------------------------------------------- #


def pnl_proxy(models, holdout, rec_interm, rec_spike, budget) -> dict:
    """Realized portfolio NPV under our policy vs legacy vs approve-all.

    Selective-labels reality: the funded holdout contains ONLY legacy-approved
    loans, so legacy == approve-all here. The lever our policy has is *declining*
    the negative-NPV loans the legacy book funded. We normalize to the hindsight
    oracle (fund exactly the realized-positive-NPV loans):

        norm(pnl) = clip((our - approve_all) / (oracle - approve_all), 0, 1)
    """
    amount = pd.to_numeric(holdout["requested_amount"], errors="coerce").to_numpy(float)
    realized = economics.realized_npv(
        holdout["default_flag"].to_numpy(float),
        pd.to_numeric(holdout["days_to_default"], errors="coerce").to_numpy(float),
        pd.to_numeric(holdout["final_recovered_amount"], errors="coerce").to_numpy(float),
        amount,
    )

    X = features.build_features(holdout)

    out = {}
    # Full ablation: {flat, timing} x {point, conservative}.
    for rule in ("flat", "timing"):
        for cons in (False, True):
            res = policy.portfolio_decisions(
                models, X, amount, rec_interm, rec_spike,
                rule=rule, conservative=cons, lo_pct=budget["enpv_lo_pct"],
            )
            d = res["decision"]
            tag = f"{rule}_{'cons' if cons else 'point'}"
            out[f"total_{tag}"] = float(np.nansum(d * realized))
            out[f"approval_rate_{tag}"] = float(np.mean(d))

    approve_all = float(np.nansum(realized))
    oracle = float(np.nansum(realized[realized > 0]))
    headline_tag = f"{config.POLICY_RULE}_{'cons' if config.POLICY_CONSERVATIVE else 'point'}"
    baseline_tag = "flat_cons"  # the pre-WS1 pipeline rule
    our_total = out[f"total_{headline_tag}"]

    denom = oracle - approve_all
    norm = (our_total - approve_all) / denom if denom > 0 else 0.0
    norm = float(np.clip(norm, 0.0, 1.0))

    out.update(
        {
            "headline_rule": headline_tag,
            "our_total": our_total,
            "legacy_total": approve_all,   # == approve-all on the funded subset
            "approve_all_total": approve_all,
            "oracle_total": oracle,
            "n_holdout": int(len(holdout)),
            "uplift_vs_legacy": float(our_total - approve_all),
            "uplift_vs_flat_baseline": float(our_total - out[f"total_{baseline_tag}"]),
            "approval_rate_headline": out[f"approval_rate_{headline_tag}"],
            "norm": norm,
        }
    )
    return out


# --------------------------------------------------------------------------- #
# Trajectory proxy (S_traj, 25%)
# --------------------------------------------------------------------------- #


def traj_proxy(models, holdout, labeled, budget) -> dict:
    """Cohort-size-weighted MAE of predicted vs realized CDR_{w,a} on the holdout.

    Realized CDR for cohort w at weekly age a = fraction of cohort-w funded loans
    that defaulted by day 7a. Predicted = mean CIF_d[a] from the hazard ensemble
    over the same loans. Skill-scored against a leakage-free naive baseline: the
    TRAIN marginal CDR curve (computed on labeled train, NOT the holdout):

        norm(traj_err) = clip(model_MAE / naive_MAE, 0, 1)   (skill vs train marginal)
    """
    cohort = data.assign_cohort_week(holdout).to_numpy(float)
    flag = holdout["default_flag"].to_numpy(float)
    days = pd.to_numeric(holdout["days_to_default"], errors="coerce").to_numpy(float)

    # Leakage-free naive predictor: marginal CDR curve from TRAIN labeled loans.
    tr_flag = labeled["default_flag"].to_numpy(float)
    tr_days = pd.to_numeric(labeled["days_to_default"], errors="coerce").to_numpy(float)

    X = features.build_features(holdout)
    # Ensemble-mean CIF per loan, shape (N, WEEKS).
    cif_stack = []
    for m in models:
        static_cols = [c for c in m.feature_cols if c != "loan_age_weeks"]
        Xa = features.align_columns(X, static_cols)
        h_d, h_p = survival.hazard_curves(m, Xa)
        cif_stack.append(survival.cif_default(h_d, h_p))
    cif = np.mean(cif_stack, axis=0)  # (N, WEEKS)

    weeks = config.WEEKS
    abs_err_w = 0.0
    n_w = 0.0
    naive_err_w = 0.0

    # Naive baseline predictor = TRAIN marginal CDR curve (no holdout peeking).
    naive_curve = np.array(
        [np.mean((tr_flag == 1) & (tr_days <= 7 * (a + 1))) for a in range(weeks)]
    )

    cells = []
    for c in np.unique(cohort[np.isfinite(cohort)]):
        sel = cohort == c
        ncoh = int(np.sum(sel))
        if ncoh == 0:
            continue
        for a in range(weeks):
            realized_cdr = float(np.mean((flag[sel] == 1) & (days[sel] <= 7 * (a + 1))))
            pred_cdr = float(np.mean(cif[sel, a]))
            abs_err_w += ncoh * abs(pred_cdr - realized_cdr)
            naive_err_w += ncoh * abs(naive_curve[a] - realized_cdr)
            n_w += ncoh
            cells.append({"cohort_week": int(c), "loan_age_weeks": a + 1,
                          "realized": realized_cdr, "pred": pred_cdr, "n": ncoh})

    model_mae = abs_err_w / n_w if n_w else float("nan")
    naive_mae = naive_err_w / n_w if n_w else float("nan")
    norm = float(np.clip(model_mae / naive_mae, 0.0, 1.0)) if naive_mae else 1.0
    return {
        "weighted_mae": float(model_mae),
        "naive_weighted_mae": float(naive_mae),
        "norm_err": norm,
        "score": float(1.0 - norm),
        "n_cohorts": int(len(np.unique(cohort[np.isfinite(cohort)]))),
    }


# --------------------------------------------------------------------------- #
# Calibration proxy (S_cal, 20%)
# --------------------------------------------------------------------------- #


def _binned_coverage(point, y, lo, hi, n_bins: int = 20) -> tuple[float, int]:
    """Group-level coverage: fraction of PD-bins whose empirical default rate
    falls within the bin's mean band. A per-loan PD band is a credible interval
    on a latent probability we never observe per loan, hence the binned test."""
    order = np.argsort(point)
    covered = used = 0
    for b in np.array_split(order, n_bins):
        if len(b) == 0:
            continue
        rate = float(np.mean(y[b]))
        used += 1
        if float(np.mean(lo[b])) <= rate <= float(np.mean(hi[b])):
            covered += 1
    return (covered / used if used else 0.0), used


def cal_proxy(models, holdout, budget, n_bins: int = 20) -> dict:
    """90% PD-interval coverage + width on the funded holdout (WS3).

    Uses split-conformal bands (calibration.fit_pd_calibrator). To avoid the
    circularity of calibrating and evaluating on the same rows, we CROSS-fit: the
    holdout is split into F folds (F = compute budget conformal_folds); each fold
    is scored with a calibrator fit on the other folds, then binned coverage and
    width are measured out-of-fold. The raw ensemble bands are reported alongside
    as the pre-WS3 baseline.
    """
    amount = pd.to_numeric(holdout["requested_amount"], errors="coerce").to_numpy(float)
    y = holdout["default_flag"].to_numpy(float)
    X = features.build_features(holdout)
    res = policy.portfolio_decisions(
        models, X, amount, rule=config.POLICY_RULE,
        conservative=config.POLICY_CONSERVATIVE, lo_pct=budget["enpv_lo_pct"],
    )
    point = res["predicted_pd"]
    n_bins = 10  # fixed granularity for calibration AND evaluation (bin-noise parity)

    # Pre-WS3 baseline: raw ensemble bands.
    raw_cov, _ = _binned_coverage(point, y, res["pd_lower_90"], res["pd_upper_90"], n_bins)
    raw_width = float(np.mean(res["pd_upper_90"] - res["pd_lower_90"]))

    # Repeated 50/50 split-conformal: calibrate on one half, evaluate held-out on
    # the other (equal-size halves -> same bin noise -> the half-width generalizes).
    # The number of random splits scales with the compute budget.
    n_splits = max(2, int(budget["conformal_folds"]))
    covs, widths = [], []
    for s in range(n_splits):
        rng = np.random.default_rng(config.RANDOM_SEED + s)
        perm = rng.permutation(len(holdout))
        a, b = perm[: len(perm) // 2], perm[len(perm) // 2:]
        for cal_idx, test_idx in ((a, b), (b, a)):
            hw = calibration.fit_pd_band(point[cal_idx], y[cal_idx], n_bins=n_bins)
            lo, pt, hi = calibration.apply_pd_band(hw, point[test_idx])
            cov, _ = _binned_coverage(pt, y[test_idx], lo, hi, n_bins=n_bins)
            covs.append(cov)
            widths.append(float(np.mean(hi - lo)))
    coverage = float(np.mean(covs))
    mean_width = float(np.mean(widths))

    coverage_term = max(0.0, 1.0 - abs(coverage - 0.90) / 0.90)
    width_term = 1.0 - float(np.clip(mean_width / 0.5, 0.0, 1.0))
    score = float(np.clip(0.7 * coverage_term + 0.3 * width_term, 0.0, 1.0))
    return {
        "coverage": coverage,
        "mean_width": mean_width,
        "raw_ensemble_coverage": float(raw_cov),
        "raw_ensemble_width": raw_width,
        "score": score,
        "mean_predicted_pd": float(np.mean(point)),
        "empirical_default_rate": float(np.mean(y)),
    }


# --------------------------------------------------------------------------- #
# Counterfactual proxy (S_C, 10%) -- harness g-computation vs naive
# --------------------------------------------------------------------------- #


def c_proxy(budget) -> dict:
    """g-comp MAE vs naive MAE on the harness strong-propagation slice (sev 0.4).

    Best-effort: requires the closed-loop-default-detection harness on PYTHONPATH.
    norm(c_gap) = clip((naive_mae - gcomp_mae) / naive_mae, 0, 1).
    """
    try:
        if str(HARNESS_SRC) not in sys.path:
            sys.path.insert(0, str(HARNESS_SRC))
        from cldd import counterfactual  # noqa: E402

        n_app = 1500 + 500 * budget["sev_seeds"]  # scales with budget
        result = counterfactual.run_counterfactual_eval(
            n_applicants=int(n_app), selection_severity=0.4,
            n_query_applicants=200, seed=config.RANDOM_SEED,
        )
        naive = float(result.naive_mae_strong_propagation)
        gcomp = float(result.gcomp_mae_strong_propagation)
        norm = float(np.clip((naive - gcomp) / naive, 0.0, 1.0)) if naive > 0 else 0.0
        return {"naive_mae": naive, "gcomp_mae": gcomp, "norm": norm, "available": True}
    except Exception as e:  # harness optional under tight scope
        return {"available": False, "error": str(e), "norm": 0.0}


# --------------------------------------------------------------------------- #
# Writeup checklist (S_write, 15%)
# --------------------------------------------------------------------------- #


def writeup_proxy() -> dict:
    """Fraction of writeup-backing artifacts that exist."""
    art = config.ARTIFACTS_DIR
    checklist = {
        "scorecard": (art / "scorecard.json").exists(),
        "pnl_backtest_fig": (art / "pnl_backtest.png").exists(),
        "coverage_curve": (art / "declined_coverage.csv").exists(),
        "compute_curve": (art / "compute_curve.csv").exists(),
        "writeup_md": (config.SUBMISSION_DIR / "submission_D_writeup.md").exists(),
    }
    frac = float(np.mean(list(checklist.values())))
    return {"checklist": checklist, "frac": frac}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def build_scorecard(level: str) -> dict:
    budget = compute_mod.budget(level)
    ds = data.load_dataset()
    labeled = ds.labeled_train
    holdout = _load_holdout(ds)

    models = _fit_models(labeled, budget["bag_size"])
    rec_interm, rec_spike = recovery.estimate_recovery_rates_by_timing(labeled)

    pnl = pnl_proxy(models, holdout, rec_interm, rec_spike, budget)
    traj = traj_proxy(models, holdout, labeled, budget)
    cal = cal_proxy(models, holdout, budget)
    c = c_proxy(budget)
    wr = writeup_proxy()

    weighted = (
        0.30 * pnl["norm"]
        + 0.25 * traj["score"]
        + 0.20 * cal["score"]
        + 0.10 * c["norm"]
        + 0.15 * wr["frac"]
    )

    return {
        "_caveat": "PROXY score; true score uses hidden labels + hidden normalization. "
        "Measured on the out-of-time validation funded holdout; weights per brief p.14.",
        "compute": level,
        "budget": dict(budget),
        "policy_rule": config.POLICY_RULE,
        "recovery_interm": rec_interm,
        "recovery_spike": rec_spike,
        "weights": {"pnl": 0.30, "traj": 0.25, "cal": 0.20, "c": 0.10, "write": 0.15},
        "pnl_proxy": pnl,
        "traj_proxy": traj,
        "cal_proxy": cal,
        "c_proxy": c,
        "writeup_proxy": wr,
        "weighted_proxy": float(weighted),
    }


def _print_table(sc: dict) -> None:
    print("\n" + "=" * 70)
    print(f"PROXY SCORECARD  (compute={sc['compute']}, rule={sc['policy_rule']})")
    print(sc["_caveat"])
    print("-" * 70)
    p = sc["pnl_proxy"]
    print(f"S_P&L  (0.30)  norm={p['norm']:.3f}  our[{p['headline_rule']}]=${p['our_total']:,.0f}  "
          f"legacy/approve-all=${p['approve_all_total']:,.0f}  oracle=${p['oracle_total']:,.0f}")
    print(f"               uplift vs flat-baseline=${p['uplift_vs_flat_baseline']:,.0f}  "
          f"approval={p['approval_rate_headline']:.3f}  "
          f"[flat_cons=${p['total_flat_cons']:,.0f} timing_point=${p['total_timing_point']:,.0f}]")
    t = sc["traj_proxy"]
    print(f"S_traj (0.25)  score={t['score']:.3f}  wMAE={t['weighted_mae']:.4f}  "
          f"(naive {t['naive_weighted_mae']:.4f})")
    cal = sc["cal_proxy"]
    print(f"S_cal  (0.20)  score={cal['score']:.3f}  coverage={cal['coverage']:.3f}  "
          f"width={cal['mean_width']:.4f}")
    c = sc["c_proxy"]
    if c.get("available"):
        print(f"S_C    (0.10)  norm={c['norm']:.3f}  naive_mae={c['naive_mae']:.4f}  "
              f"gcomp_mae={c['gcomp_mae']:.4f}")
    else:
        print(f"S_C    (0.10)  norm={c['norm']:.3f}  (harness unavailable: best-effort)")
    w = sc["writeup_proxy"]
    print(f"S_write(0.15)  frac={w['frac']:.3f}  {w['checklist']}")
    print("-" * 70)
    print(f"WEIGHTED PROXY = {sc['weighted_proxy']:.4f}")
    print("=" * 70 + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compute", choices=compute_mod.LEVELS, default="med")
    ap.add_argument("--out", default=str(config.ARTIFACTS_DIR / "scorecard.json"))
    args = ap.parse_args(argv)

    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    sc = build_scorecard(args.compute)
    payload = json.dumps(sc, indent=2)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(payload)
    # Mirror into the committed reports/ dir (artifacts/ is gitignored).
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / Path(args.out).name).write_text(payload)
    _print_table(sc)
    print(f"wrote {args.out}  (+ reports/{Path(args.out).name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

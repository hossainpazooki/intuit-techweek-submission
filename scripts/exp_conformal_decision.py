"""Out-of-fold adopt-gate for the WS3->decision lever (conformal PD in the E[NPV] rule).

Question: does re-pricing the approve/decline E[NPV] at the conformal UPPER PD
bound (policy.conformal_adjusted_enpv, config.POLICY_CONFORMAL_DECISION) improve
REALIZED portfolio P&L on the out-of-time validation funded holdout, measured
out-of-fold?

Protocol (no in-fold leakage):
  - The hazard ensemble is fit on TRAIN labeled loans only (same as the pipeline).
  - The holdout (validation funded, has outcomes) is split 50/50, n_splits times
    (rng seeded config.RANDOM_SEED + s, identical to cal_proxy's pattern). For
    each (cal, test) ordering:
      * the split-conformal half-width is fit on the CAL half only
        (calibration.fit_pd_band, same defaults as the pipeline);
      * BASELINE arm  = current headline decision (config.POLICY_RULE timing,
        point E[NPV]), which uses nothing from the holdout, evaluated on TEST;
      * CANDIDATE arm = decision at the conformal-upper-bound E[NPV] with the
        CAL-half half-width, evaluated on TEST.
    Realized P&L = sum over TEST of decision_i * realized_npv_i.
  - Paired diffs (candidate - baseline) are reported as mean +- sd over the
    2 * n_splits evaluations.

Also asserts: (a) the fast per-split path equals a full
policy.portfolio_decisions(conformal_hw=...) call (wiring check), and (b)
re-running the candidate decision with the same inputs is byte-identical
(determinism).

Usage:
    python scripts/exp_conformal_decision.py --compute med --splits 10
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compute", choices=compute_mod.LEVELS, default="med")
    ap.add_argument("--splits", type=int, default=10)
    ap.add_argument(
        "--out", default=str(config.ARTIFACTS_DIR / "exp_conformal_decision.json")
    )
    args = ap.parse_args(argv)

    budget = compute_mod.budget(args.compute)
    ds = data.load_dataset()
    labeled = ds.labeled_train
    holdout = ds.validation[ds.validation["default_flag"].notna()].reset_index(drop=True)
    n = len(holdout)
    print(f"holdout = validation funded, n={n}  compute={args.compute}  "
          f"bag={budget['bag_size']}  splits={args.splits} (x2 orderings)")

    seeds = [config.RANDOM_SEED + i for i in range(budget["bag_size"])]
    models = survival.fit_hazard_ensemble(labeled, seeds)
    rec_interm, rec_spike = recovery.estimate_recovery_rates_by_timing(labeled)

    amount = pd.to_numeric(holdout["requested_amount"], errors="coerce").to_numpy(float)
    y = holdout["default_flag"].to_numpy(float)
    realized = economics.realized_npv(
        y,
        pd.to_numeric(holdout["days_to_default"], errors="coerce").to_numpy(float),
        pd.to_numeric(holdout["final_recovered_amount"], errors="coerce").to_numpy(float),
        amount,
    )
    X = features.build_features(holdout)

    # Score the full holdout ONCE. All per-loan quantities are split-independent
    # (the ensemble never sees the holdout); only the conformal half-width is
    # split-dependent.
    res = policy.portfolio_decisions(
        models, X, amount, rec_interm, rec_spike,
        rule=config.POLICY_RULE, conservative=config.POLICY_CONSERVATIVE,
        lo_pct=budget["enpv_lo_pct"],
    )
    d_base = res["decision"]
    pd_point = res["predicted_pd"]
    enpv_point = res["enpv_point"]

    # Ensemble-mean per-week default mass (timing profile) for the fast path.
    p_def_stack = []
    for m in models:
        static_cols = [c for c in m.feature_cols if c != "loan_age_weeks"]
        Xa = features.align_columns(X, static_cols)
        p_def, _ = survival.default_week_probs(m, Xa)
        p_def_stack.append(p_def)
    p_def_mean = np.mean(np.stack(p_def_stack), axis=0)

    def candidate_decisions(idx: np.ndarray, hw: float) -> np.ndarray:
        enpv_conf = policy.conformal_adjusted_enpv(
            enpv_point[idx], pd_point[idx], p_def_mean[idx], amount[idx],
            hw, rec_interm, rec_spike,
        )
        d = (enpv_conf > 0).astype(int)
        return np.where(np.isfinite(amount[idx]), d, 0).astype(int)

    rows = []
    wiring_checked = False
    for s in range(args.splits):
        rng = np.random.default_rng(config.RANDOM_SEED + s)
        perm = rng.permutation(n)
        a, b = perm[: n // 2], perm[n // 2:]
        for tag, (cal_idx, test_idx) in (("ab", (a, b)), ("ba", (b, a))):
            hw = calibration.fit_pd_band(pd_point[cal_idx], y[cal_idx])
            d_cand = candidate_decisions(test_idx, hw)

            if not wiring_checked:
                # Wiring check: fast path == full portfolio_decisions(conformal_hw=).
                res_full = policy.portfolio_decisions(
                    models, X.iloc[test_idx], amount[test_idx], rec_interm, rec_spike,
                    rule=config.POLICY_RULE, conservative=config.POLICY_CONSERVATIVE,
                    conformal_hw=hw, lo_pct=budget["enpv_lo_pct"],
                )
                assert np.array_equal(res_full["decision"], d_cand), "wiring mismatch"
                # Determinism: same inputs -> identical decisions.
                assert np.array_equal(candidate_decisions(test_idx, hw), d_cand)
                wiring_checked = True
                print("wiring + determinism checks: OK")

            pnl_b = float(np.nansum(d_base[test_idx] * realized[test_idx]))
            pnl_c = float(np.nansum(d_cand * realized[test_idx]))
            rows.append(
                {
                    "split": s, "ordering": tag, "hw": float(hw),
                    "pnl_base": pnl_b, "pnl_cand": pnl_c, "diff": pnl_c - pnl_b,
                    "appr_base": float(np.mean(d_base[test_idx])),
                    "appr_cand": float(np.mean(d_cand)),
                }
            )

    df = pd.DataFrame(rows)
    summary = {
        "compute": args.compute,
        "n_holdout": int(n),
        "n_evals": int(len(df)),
        "headline_rule": f"{config.POLICY_RULE}_{'cons' if config.POLICY_CONSERVATIVE else 'point'}",
        "hw_mean": float(df["hw"].mean()),
        "hw_sd": float(df["hw"].std(ddof=1)),
        "pnl_base_mean": float(df["pnl_base"].mean()),
        "pnl_base_sd": float(df["pnl_base"].std(ddof=1)),
        "pnl_cand_mean": float(df["pnl_cand"].mean()),
        "pnl_cand_sd": float(df["pnl_cand"].std(ddof=1)),
        "diff_mean": float(df["diff"].mean()),
        "diff_sd": float(df["diff"].std(ddof=1)),
        "n_cand_wins": int((df["diff"] > 0).sum()),
        "appr_base_mean": float(df["appr_base"].mean()),
        "appr_cand_mean": float(df["appr_cand"].mean()),
        "full_holdout_pnl_base": float(np.nansum(d_base * realized)),
        "adopt_gate_pnl_improves": bool(df["diff"].mean() > 0),
        "rows": rows,
    }

    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2))

    print("\n--- OOF realized P&L on the funded holdout (per test half) ---")
    print(f"half-width hw       = {summary['hw_mean']:.4f} +- {summary['hw_sd']:.4f}")
    print(f"BASELINE  ({summary['headline_rule']}): "
          f"${summary['pnl_base_mean']:,.0f} +- ${summary['pnl_base_sd']:,.0f}  "
          f"(approval {summary['appr_base_mean']:.3f})")
    print(f"CANDIDATE (conformal-upper E[NPV]):    "
          f"${summary['pnl_cand_mean']:,.0f} +- ${summary['pnl_cand_sd']:,.0f}  "
          f"(approval {summary['appr_cand_mean']:.3f})")
    print(f"paired diff (cand - base) = ${summary['diff_mean']:,.0f} "
          f"+- ${summary['diff_sd']:,.0f}   cand wins {summary['n_cand_wins']}/{summary['n_evals']}")
    print(f"ADOPT GATE (mean diff > 0): {'PASS' if summary['adopt_gate_pnl_improves'] else 'FAIL'}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

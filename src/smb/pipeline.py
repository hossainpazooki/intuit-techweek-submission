"""End-to-end orchestration: data -> models -> the four submission files.

`python -m smb.pipeline` (or scripts/run_all.py) regenerates
submission/submission_{A,B,C}.csv from scratch, reproducibly. Deliverable D is
written by hand from submission_D_writeup_template.md.

The deliverables are coupled: B and C both depend on the hazard ensemble used for
A, and B's cohort curves are conditioned on A's approve/decline decisions.

Assembly (guaranteed-PASS):
  A  LEFT-JOIN onto expected_ids/applicant_ids.txt order; decision int {0,1};
     predicted_pd/pd_lower_90/pd_upper_90 float in [0,1], non-null even for
     declines/missing (gaps filled with population-mean pd + decision 0);
     clip_and_order on (lower, point, upper).
  B  169 rows from the template grid; cdr cols overwritten from the hazard
     trajectory; per-cohort monotone; clip_and_order; no NaN.
  C  LEFT-JOIN onto expected_ids/query_ids.txt order; clip_and_order; gaps filled
     with pop-mean + wide band; no NaN.

Then the official validator is run in-process and report.passed is asserted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import calibration, causal, config, dag, data, features, policy, survival


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _read_id_order(path: Path) -> list[str]:
    """Read an expected_ids/*.txt file preserving order, dropping blanks."""
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def _ensemble_lifetime_pd(models, X_static: pd.DataFrame) -> np.ndarray:
    """Per-model lifetime PD; returns shape (n_models, n_rows)."""
    rows = []
    for model in models:
        static_cols = [c for c in model.feature_cols if c != "loan_age_weeks"]
        X_aligned = features.align_columns(X_static, static_cols)
        rows.append(np.asarray(survival.lifetime_pd(model, X_aligned), dtype=float).ravel())
    return np.vstack(rows)


# --------------------------------------------------------------------------- #
# Deliverable A
# --------------------------------------------------------------------------- #


def build_submission_a(
    ds: data.Dataset,
    models,
    recovery_rate: float,
) -> pd.DataFrame:
    """columns: applicant_id, decision, predicted_pd, pd_lower_90, pd_upper_90.

    Scores the decision frame with the hazard ensemble (lifetime PD per model ->
    ensemble point + 5/95 bounds), applies the conservative upper-bound NPV
    decision rule, then LEFT-JOINs onto the expected applicant_id order. Any
    missing applicant gets the population-mean PD + decision 0.
    """
    df = ds.decision_frame.reset_index(drop=True)

    X = features.build_features(df)
    samples = _ensemble_lifetime_pd(models, X)  # (n_models, n)
    lower, point, upper = calibration.ensemble_intervals(
        samples, lo=config.INTERVAL_LOWER_Q, hi=config.INTERVAL_UPPER_Q
    )

    amount = pd.to_numeric(df["requested_amount"], errors="coerce").to_numpy(dtype=float)
    # Decide on the conservative UPPER PD bound.
    decision = policy.decide(upper, amount, recovery_rate)
    # A loan with an unusable amount cannot be priced -> decline.
    decision = np.where(np.isfinite(amount), decision, 0).astype(int)

    scored = pd.DataFrame(
        {
            "applicant_id": df["applicant_id"].astype(str).to_numpy(),
            "decision": decision,
            "predicted_pd": point,
            "pd_lower_90": lower,
            "pd_upper_90": upper,
        }
    )

    # LEFT-JOIN onto the canonical id order.
    order = _read_id_order(config.EXPECTED_IDS_DIR / "applicant_ids.txt")
    out = pd.DataFrame({"applicant_id": order}).merge(
        scored, on="applicant_id", how="left"
    )

    # Fill gaps (ids not scored) with population mean + decision 0.
    pop_pd = float(np.nanmean(point)) if np.isfinite(np.nanmean(point)) else 0.5
    out["decision"] = out["decision"].fillna(0).astype(int)
    out["predicted_pd"] = out["predicted_pd"].fillna(pop_pd)
    out["pd_lower_90"] = out["pd_lower_90"].fillna(pop_pd)
    out["pd_upper_90"] = out["pd_upper_90"].fillna(pop_pd)

    lo, pt, hi = calibration.clip_and_order(
        out["pd_lower_90"].to_numpy(dtype=float),
        out["predicted_pd"].to_numpy(dtype=float),
        out["pd_upper_90"].to_numpy(dtype=float),
    )
    out["pd_lower_90"] = lo
    out["predicted_pd"] = pt
    out["pd_upper_90"] = hi

    return out[["applicant_id", "decision", "predicted_pd", "pd_lower_90", "pd_upper_90"]]


# --------------------------------------------------------------------------- #
# Deliverable B
# --------------------------------------------------------------------------- #


def build_submission_b(
    ds: data.Dataset,
    models,
    sub_a: pd.DataFrame,
) -> pd.DataFrame:
    """columns: cohort_week, loan_age_weeks, cumulative_default_rate, cdr_*_90.

    Builds the hazard trajectory for approved cohort loans, then writes it onto
    the template's 13x13 integer grid (per-cohort monotone, clip_and_order).
    """
    df = ds.decision_frame.reset_index(drop=True)

    cohort_weeks = data.assign_cohort_week(df).to_numpy(dtype=float)

    # Map A's decisions back onto the decision-frame row order via applicant_id.
    dec_map = dict(zip(sub_a["applicant_id"].astype(str), sub_a["decision"].astype(int)))
    decisions = df["applicant_id"].astype(str).map(dec_map).fillna(0).astype(int).to_numpy()

    traj = survival.predict_trajectory(models, df, decisions, cohort_weeks)

    # Anchor on the template grid to guarantee the exact 13x13 row set/order.
    template = pd.read_csv(config.SUBMISSION_B_TEMPLATE_CSV)
    grid = template[["cohort_week", "loan_age_weeks"]].copy()
    grid["cohort_week"] = grid["cohort_week"].astype(int)
    grid["loan_age_weeks"] = grid["loan_age_weeks"].astype(int)

    traj = traj.copy()
    traj["cohort_week"] = traj["cohort_week"].astype(int)
    traj["loan_age_weeks"] = traj["loan_age_weeks"].astype(int)

    out = grid.merge(traj, on=["cohort_week", "loan_age_weeks"], how="left")

    # Any unfilled cell -> 0.0 (defensive; predict_trajectory covers all cells).
    for col in ("cumulative_default_rate", "cdr_lower_90", "cdr_upper_90"):
        out[col] = out[col].fillna(0.0)

    # Re-enforce per-cohort monotonicity + ordering after the merge.
    out = out.sort_values(["cohort_week", "loan_age_weeks"]).reset_index(drop=True)
    for w, g in out.groupby("cohort_week"):
        idx = g.index
        point = survival.enforce_monotone(g["cumulative_default_rate"].to_numpy(dtype=float))
        lower = survival.enforce_monotone(g["cdr_lower_90"].to_numpy(dtype=float))
        upper = survival.enforce_monotone(g["cdr_upper_90"].to_numpy(dtype=float))
        lower, point, upper = calibration.clip_and_order(lower, point, upper)
        out.loc[idx, "cumulative_default_rate"] = point
        out.loc[idx, "cdr_lower_90"] = lower
        out.loc[idx, "cdr_upper_90"] = upper

    return out[
        [
            "cohort_week",
            "loan_age_weeks",
            "cumulative_default_rate",
            "cdr_lower_90",
            "cdr_upper_90",
        ]
    ]


# --------------------------------------------------------------------------- #
# Deliverable C
# --------------------------------------------------------------------------- #


def build_submission_c(
    ds: data.Dataset,
    models,
    dag_children: dict | None = None,
) -> pd.DataFrame:
    """columns: query_id, predicted_pd_cf, pd_cf_lower_90, pd_cf_upper_90."""
    if dag_children is None:
        dag_children = dag.intervenable_dag()

    queries = causal.load_queries()
    df = ds.decision_frame.reset_index(drop=True)

    cf = causal.estimate_counterfactual(models, df, queries, dag_children)
    cf["query_id"] = cf["query_id"].astype(str)

    # LEFT-JOIN onto canonical query id order.
    order = _read_id_order(config.EXPECTED_IDS_DIR / "query_ids.txt")
    out = pd.DataFrame({"query_id": order}).merge(cf, on="query_id", how="left")

    # Fill any gap with a wide population band.
    pop_point = float(np.nanmean(cf["predicted_pd_cf"].to_numpy(dtype=float)))
    if not np.isfinite(pop_point):
        pop_point = 0.5
    out["predicted_pd_cf"] = out["predicted_pd_cf"].fillna(pop_point)
    out["pd_cf_lower_90"] = out["pd_cf_lower_90"].fillna(max(0.0, pop_point - 0.25))
    out["pd_cf_upper_90"] = out["pd_cf_upper_90"].fillna(min(1.0, pop_point + 0.25))

    lo, pt, hi = calibration.clip_and_order(
        out["pd_cf_lower_90"].to_numpy(dtype=float),
        out["predicted_pd_cf"].to_numpy(dtype=float),
        out["pd_cf_upper_90"].to_numpy(dtype=float),
    )
    out["pd_cf_lower_90"] = lo
    out["predicted_pd_cf"] = pt
    out["pd_cf_upper_90"] = hi

    return out[["query_id", "predicted_pd_cf", "pd_cf_lower_90", "pd_cf_upper_90"]]


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def main() -> None:
    from . import recovery

    config.SUBMISSION_DIR.mkdir(exist_ok=True)

    print("[1/6] Loading data...")
    ds = data.load_dataset()
    labeled = ds.labeled_train
    print(
        f"      train={len(ds.train)} labeled={len(labeled)} "
        f"decision_frame={len(ds.decision_frame)}"
    )

    print("[2/6] Asserting no censoring in labeled set...")
    survival.fit_hazard_model  # touch to ensure import
    from . import survival_data

    survival_data.assert_no_censoring(labeled)

    seeds = [config.RANDOM_SEED + i for i in range(config.N_BAG_SEEDS)]
    print(f"[3/6] Fitting hazard ensemble ({len(seeds)} seeds)...")
    models = survival.fit_hazard_ensemble(labeled, seeds)

    print("[4/6] Estimating recovery rate...")
    recovery_rate = recovery.estimate_recovery_rate(labeled)
    print(f"      recovery_rate={recovery_rate:.4f}")

    print("[5/6] Building submissions A/B/C...")
    sub_a = build_submission_a(ds, models, recovery_rate)
    sub_a.to_csv(config.FILE_A, index=False)

    sub_b = build_submission_b(ds, models, sub_a)
    sub_b.to_csv(config.FILE_B, index=False)

    sub_c = build_submission_c(ds, models)
    sub_c.to_csv(config.FILE_C, index=False)

    # --- Sanity values (reported, not gating).
    mean_pd = float(sub_a["predicted_pd"].mean())
    approval_rate = float((sub_a["decision"] == 1).mean())
    b_age13 = sub_b[sub_b["loan_age_weeks"] == 13]["cumulative_default_rate"].mean()
    print(
        f"      mean predicted_pd={mean_pd:.4f}  approval_rate={approval_rate:.4f}  "
        f"B cdr@age13(mean)={float(b_age13):.4f}"
    )
    width_a = float((sub_a["pd_upper_90"] - sub_a["pd_lower_90"]).mean())
    width_c = float((sub_c["pd_cf_upper_90"] - sub_c["pd_cf_lower_90"]).mean())
    print(f"      mean band width  A={width_a:.4f}  C={width_c:.4f}")

    print(f"[6/6] Validating submission in-process at {config.SUBMISSION_DIR}...")
    _validate_in_process()


def _validate_in_process() -> None:
    """Import the repo-root validator and assert PASS, printing the report."""
    root = config.ROOT
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import validate_submission as vs

    spec = vs.ExpectedSpec.from_manifest(config.EXPECTED_IDS_DIR)
    report = vs.validate_submission(config.SUBMISSION_DIR, spec)
    vs._print_report(report)
    assert report.passed, "Submission validator reported errors (see above)."


if __name__ == "__main__":
    main()

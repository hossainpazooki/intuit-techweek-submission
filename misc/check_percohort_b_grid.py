"""One-off check: the per-cohort recalibration path yields a validator-clean B grid.

Forces ARTIFICIALLY distinct per-cohort factors (linspace 0.85..1.45 -- a much
harsher spread than the EB estimator would ever emit, since tau^2=0 shrinks all
factors to the global ratio) through survival.predict_trajectory and asserts the
validator's B constraints: exact 13x13 grid, values in [0,1], lower<=point<=upper,
and non-decreasing in age within each cohort. Run from repo root:

    python misc/check_percohort_b_grid.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smb import config, data, survival  # noqa: E402

ds = data.load_dataset()
labeled = ds.labeled_train
models = survival.fit_hazard_ensemble(labeled, [config.RANDOM_SEED + i for i in range(3)])

df = ds.decision_frame.reset_index(drop=True)
cohort_weeks = data.assign_cohort_week(df).to_numpy(dtype=float)
decisions = np.ones(len(df), dtype=int)  # approve-all: every cohort populated

factors = np.linspace(0.85, 1.45, config.N_COHORT_WEEKS)
traj = survival.predict_trajectory(
    models, df, decisions, cohort_weeks,
    cif_scale=1.1, cif_scale_by_cohort=factors,
)

assert len(traj) == 169, f"rows {len(traj)} != 169"
assert sorted(traj["cohort_week"].unique()) == list(range(1, 14))
assert sorted(traj["loan_age_weeks"].unique()) == list(range(1, 14))
for col in ("cumulative_default_rate", "cdr_lower_90", "cdr_upper_90"):
    v = traj[col].to_numpy(float)
    assert np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all(), col
assert (traj["cdr_lower_90"] <= traj["cumulative_default_rate"] + 1e-12).all()
assert (traj["cumulative_default_rate"] <= traj["cdr_upper_90"] + 1e-12).all()
for c, g in traj.sort_values(["cohort_week", "loan_age_weeks"]).groupby("cohort_week"):
    for col in ("cumulative_default_rate", "cdr_lower_90", "cdr_upper_90"):
        v = g[col].to_numpy(float)
        assert (np.diff(v) >= -1e-12).all(), f"cohort {c} {col} not monotone"

# Factors actually moved the cohorts apart (the scaling is live, not a no-op).
age13 = traj[traj["loan_age_weeks"] == 13].sort_values("cohort_week")
spread = float(age13["cumulative_default_rate"].max() - age13["cumulative_default_rate"].min())
print(f"age-13 CDR by cohort: {age13['cumulative_default_rate'].round(4).tolist()}")
print(f"spread = {spread:.4f} (factors 0.85..1.45 applied)")
assert spread > 0.02, "per-cohort scaling appears inert"
print("B-GRID CHECK: OK (169 rows, monotone per cohort, ordered, in [0,1])")

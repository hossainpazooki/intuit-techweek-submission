"""End-to-end orchestration: data -> models -> the four submission files.

`python -m smb.pipeline` (or scripts/run_all.py) should regenerate
submission/submission_{A,B,C}.csv from scratch, reproducibly. Deliverable D is
written by hand from submission_D_writeup_template.md.

The deliverables are coupled: B and C both depend on the PD model from A, and
B's cohort curves are conditioned on A's approve/decline decisions.
"""

from __future__ import annotations

import pandas as pd

from . import calibration, causal, config, data, features, model_pd, policy, survival


def build_submission_a(ds: data.Dataset) -> pd.DataFrame:
    """columns: applicant_id, decision, predicted_pd, pd_lower_90, pd_upper_90."""
    raise NotImplementedError


def build_submission_b(ds: data.Dataset, decisions: pd.DataFrame) -> pd.DataFrame:
    """columns: cohort_week, loan_age_weeks, cumulative_default_rate, cdr_*_90."""
    raise NotImplementedError


def build_submission_c(ds: data.Dataset) -> pd.DataFrame:
    """columns: query_id, predicted_pd_cf, pd_cf_lower_90, pd_cf_upper_90."""
    raise NotImplementedError


def main() -> None:
    config.SUBMISSION_DIR.mkdir(exist_ok=True)
    ds = data.load_dataset()

    sub_a = build_submission_a(ds)
    sub_a.to_csv(config.FILE_A, index=False)

    sub_b = build_submission_b(ds, sub_a)
    sub_b.to_csv(config.FILE_B, index=False)

    sub_c = build_submission_c(ds)
    sub_c.to_csv(config.FILE_C, index=False)

    print(f"Wrote A/B/C to {config.SUBMISSION_DIR}. Now run scripts/validate.py.")


if __name__ == "__main__":
    main()

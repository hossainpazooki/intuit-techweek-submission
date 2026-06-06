# SMB Underwriting Challenge — Solution

Our solution to the Intuit TechWeek SMB Underwriting Challenge. The challenge
brief and dataset are upstream Intuit material (see `README.md`,
`dataset/README.md`); this file documents *our* approach and how to reproduce it.

## Quick start

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows; use bin/activate on *nix
pip install -r requirements-dev.txt

unzip dataset/dataset-compressed.zip -d dataset/    # -> train/validation/test.csv

python scripts/eda.py        # structural facts about the data
python scripts/run_all.py    # build submission/{A,B,C}.csv + validate
python scripts/validate.py   # re-run Intuit's gate; must print PASS
```

## What the data is (drives every modeling choice)

- **Temporal split.** train = Jan 2024–Jun 2025 (the past); validation + test =
  the 13 weekly cohorts Jun 30–Sep 28 2025 (the forecast window). We decide on
  validation + test = **13,306** applicants for Deliverable A.
- **Selective labels.** Outcomes exist *only* for prior-approved, matured loans
  (51,722 of 85,340 train rows; ~17.4% default rate). We must score applicants
  the old system declined and never observed — the core bias to correct for.
- **Survival, not classification (B).** Default is a 3–90 day timing process
  (median 37 days). Deliverable B is a cumulative-hazard curve per cohort.
- **Structural missingness.** Bank-feed columns null ~36% (no linked feed);
  several columns null by construction. Missingness is signal — model it.
- **Causal vs predictive (C).** `do(feature=value)` ≠ conditioning. Only
  `intervenable=True` features (per the data dictionary) are intervened on.

## Layout

```
src/smb/
  config.py       paths, loan economics (60d term, 35% APR, 3% fee), cohort grid
  data.py         load/clean, label construction, selective-labels handling
  features.py     feature engineering + intervenable-feature metadata
  model_pd.py     calibrated PD model (A) + selection-bias correction
  policy.py       expected-profit approve/decline rule (A)
  survival.py     cohort x loan-age cumulative-default forecast (B)
  causal.py       do(feature=value) counterfactuals (C)
  calibration.py  shared 90% prediction intervals (A/B/C)
  pipeline.py     orchestrates -> submission/*.csv
scripts/
  eda.py          reproducible exploration
  run_all.py      build all deliverables + validate
  validate.py     wrapper around Intuit's validate_submission.py
submission/       the four output files (flat, exact names)
```

## Deliverable status

- [ ] A — decisions (PD model + expected-profit policy + 90% intervals)
- [ ] B — trajectory (cohort × loan-age cumulative default, monotone)
- [ ] C — counterfactuals (interventional PD)
- [ ] D — writeup (`submission_D_writeup_template.md` → PDF; §3 causal weighted most)

## Remotes

- `origin`  → private repo (our work)
- `upstream` → `intuit/intuit-techweek-nyc-hackathon-2026` (official; pull updates only)

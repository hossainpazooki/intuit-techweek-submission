# SMB Underwriting Challenge — Solution

Our solution to the Intuit TechWeek SMB Underwriting Challenge. The challenge
brief and dataset are upstream Intuit material (see `README.md`,
`dataset/README.md`); this file documents *our* approach and how to reproduce it.

> **Methodology of record: [`METHODOLOGY.md`](METHODOLOGY.md).** It is written to be
> judge-ready and honest — it separates what runs today (exploration + guardrails)
> from what is *proposed* (baseline) vs *aspirational* (research stack), and states
> the identification limits of the problem. Read it for the full technical argument;
> this file is the short orientation + run instructions.

**Honest status in one line:** this repo is today an *observability / validation
layer plus a validated scaffold* — it verifies the empirical premises and gates
submissions against the official validator. The modeling pipeline is designed and
documented but **not yet implemented** (the modeling modules are stubs).

## Quick start

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows; use bin/activate on *nix
pip install -r requirements-dev.txt

unzip dataset/dataset-compressed.zip -d dataset/    # -> train/validation/test.csv

python scripts/eda.py        # structural facts about the data
python scripts/run_all.py    # build submission/{A,B,C}.csv + validate
python scripts/validate.py   # re-run Intuit's gate; must print PASS
```

## What the data is (verified — drives every modeling choice)

- **Temporal split.** train = Jan 2024–Jun 2025 (the past); validation + test =
  the 13 weekly cohorts Jun 30–Sep 28 2025 (the forecast window). We decide on
  validation + test = **13,306** applicants for Deliverable A.
- **Selective labels with a *deterministic* funding rule.** Outcomes exist *only*
  for funded, matured loans (51,722 of 85,340 train rows; **17.45%** default rate).
  The legacy funding decision is a **perfect threshold** on `prior_underwriter_score`
  (`prior_decision==1 iff score>=~0.273`, zero mismatches, no overlap) — so the
  funding propensity is degenerate and **positivity fails globally**. IPW / DR are
  *not identified* for declined applicants; we bound or abstain rather than reweight.
- **Default timing has a point mass.** Paid loans repay at **exactly day 60**;
  defaults span days 3–60 and then a **spike at exactly day 90** (~22.5%), with none
  in between. Interest accrues over the 60-day term; default is observed through day 90
  — the NPV must separate the two horizons.
- **Informative missingness.** Bank-feed block is null **iff**
  `has_linked_bank_feed==False` (~36% of train); `days_since_*` nulls mean "no prior event."
  Missingness is signal — keep NaNs + indicators, don't blind-impute.
- **Causal vs predictive (C).** `do(feature=value)` ≠ conditioning; a one-feature
  re-score (or a SHAP value) is *observational*. We propagate structural features
  (bank-feed block, engineered ratios) and report everything else as observational
  with the gap surfaced.

## Decision rule (corrected)

Expected NPV is **strictly decreasing** in default probability (a repaid loan returns
~8.75% of principal under the full-principal convention; a default destroys most of
it). So we **approve a loan only if its expected NPV stays positive at the UPPER
confidence bound of the predicted PD** — stress-testing each loan against its most
pessimistic credible default rate, so uncertain loans correctly default to *denial*.
(Using the *lower* bound would be backwards: it is the optimistic case and would make
ignorance default to approval.) Break-even PD ≈ **8.0–8.9%** (full-principal
convention) vs the 17.45% funded-book default rate. See `METHODOLOGY.md` §3.

## Layout

Implemented = runs today; *stub* = designed/documented but raises
`NotImplementedError` (see `METHODOLOGY.md` §10).

```
src/smb/
  config.py       loan economics + cohort constants            [implemented]
  calibration.py  clip_and_order, coverage (guardrails)        [partial: helpers only]
  survival.py     enforce_monotone (cummax guardrail)          [partial: helper only]
  data.py         load/clean, label construction               [stub]
  features.py     feature engineering + intervenable metadata   [stub]
  model_pd.py     calibrated PD model (A) + selection weights   [stub]
  policy.py       NPV approve/decline rule (A)                  [stub]
  causal.py       counterfactuals (C)                           [stub]
  pipeline.py     orchestrates -> submission/*.csv              [stub]
scripts/
  eda.py          reproducible exploration / premise validation [implemented]
  validate.py     wrapper around Intuit's validate_submission.py [implemented]
  run_all.py      build all deliverables + validate             [implemented wrapper;
                                                                 blocked by stubs]
submission/       the four output files (flat, exact names)
```

## Deliverable status

Methodology validated and corrected (see `METHODOLOGY.md`); modeling not yet built.

- [x] **Premises validated** — `scripts/eda.py` (selective labels, deterministic
      funding rule, timing, missingness, overlap, query structure, cohort sizes).
- [ ] A — decisions (calibrated PD + NPV rule at the **upper** PD bound + intervals).
- [ ] B — trajectory (cohort × loan-age cumulative default, monotone by construction).
- [ ] C — counterfactuals (structural propagation; observational gap surfaced).
- [ ] D — writeup (`submission_D_writeup_template.md` → PDF; §3 causal weighted most).

## Remotes

- `origin`  → private repo (our work)
- `upstream` → `intuit/intuit-techweek-nyc-hackathon-2026` (official; pull updates only)

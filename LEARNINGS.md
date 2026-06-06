# Learnings

Non-obvious patterns, gotchas, and operational knowledge for this repo
(SMB Underwriting Challenge). Complements `CLAUDE.md` (preferences, commands) and
`METHODOLOGY.md` (the methodology of record) with the deeper insights that bit us
while building it. Every empirical number here was recomputed from the data
(`scripts/eda.py`); the technical claims were adversarially verified
(`docs/VERIFICATION.md`).

---

## 1. The submission validator is the law (and it has traps)

- **NaN is rejected everywhere — even for declines and empty B cells.** `predicted_pd`
  is required for *every* applicant including declines; B cells must all be filled.
  The pipeline guarantees this by LEFT-JOINing onto `expected_ids/*.txt` and filling
  any gap (pop-mean PD + decision 0; overall-mean curve for empty cohorts).
- **Build by LEFT-JOIN onto the shipped ID lists**, not from the data. The validator
  reads `expected_ids/applicant_ids.txt` (13,306), `query_ids.txt` (900), and
  `manifest.json` (`n_cohort_weeks=13`) — not the CSVs. Match those exactly or you
  get `missing_ids` / `unknown_ids` errors.
- **B must be the full 13×13 *integer* grid (169 rows) and non-decreasing in age
  within each cohort** (tol 1e-9). Anchor on `dataset/submission_B_template.csv` and
  overwrite only the three prediction columns.
- **Columns are matched by name, not order; extra columns only warn.** Missing D PDF
  is a WARN, not an error — A/B/C are the hard gate.
- **Always end a run by validating in-process** (`pipeline.main()` calls
  `validate_submission.validate_submission` and asserts `report.passed`). Don't trust
  that the CSVs are valid because the code "looks right."

## 2. Data realities (these drive every design choice)

- **Selective labels.** Outcomes exist only on funded+matured rows
  (`default_flag.notna()`; 51,722 of 85,340 train). Declined and all test rows are
  blank. Train your outcome model on `ds.labeled_train`, never the full frame.
- **The legacy funding rule is DETERMINISTIC.** `prior_decision == 1` iff
  `prior_underwriter_score >= ~0.273`, with **zero** mismatches and *no overlap*
  (max declined 0.27297 < min approved 0.27301). Consequence: funding propensity is
  degenerate → **positivity fails globally** → IPW / doubly-robust are *not
  identified*. `src/smb/propensity.py` exists for diagnostics only; the hazard model
  is fit **unweighted**. Do not "add IPW to improve it" — it's mathematically unsound
  here.
- **`prior_decision` is constant (`==1`) in the labeled set** → it's excluded from
  features (no variance, pure selection signal). `prior_approved_amount` is also
  excluded (selection leakage).
- **Timing has a point mass, not a smooth tail.** Paid loans repay at *exactly* day
  60; defaults span days 3–60 and then jump to a spike at *exactly day 90* (22.5% of
  defaults), with **zero** in the open interval (60, 90). Interest accrues over 60
  days; default is observed through 90 — keep the two horizons separate.
- **Bank-feed missingness is structural, not random.** The 6 bank-feed columns are
  null *iff* `has_linked_bank_feed == False`. Preserve the NaN + add `*_was_null`
  indicators; never blind-impute.
- **Validation is only partially labeled.** Only its funded subset (2,551 / 4,489)
  has outcomes, so calibration on validation is itself conditioned on the legacy
  policy. Don't treat validation as a clean labeled holdout.

## 3. Environment & stack (Python 3.14 on Windows)

- **sklearn-only by necessity.** Python 3.14 has no wheels for
  lightgbm/xgboost/lifelines/econml/doubleml. The flexible learner is
  `HistGradientBoostingClassifier` (native NaN handling + `sample_weight`); isotonic
  via `IsotonicRegression`; everything else hand-rolled. If a method needs the missing
  libs, it's `[FUTURE]`, not a `pip install`.
- **Windows console is cp1252.** Always run `PYTHONUTF8=1 python ...` and keep
  `print()` ASCII-only — unicode like `π`, `≈`, `≤`, `∩` crashes with
  `UnicodeEncodeError`. Unicode in `.md` files is fine; in stdout it is not.
- **Keep model features numeric (no `category` dtype).** Integer-coded categoricals
  are treated as numeric/ordinal. This is a deliberate robustness call: pandas
  `category` dtype causes cross-split category-alignment bugs between
  train / decision-frame / counterfactual frames. `build_features` returns all-float64
  with a stable, sorted column order; `align_columns` reindexes other frames to match.

## 4. The hazard spine — design tricks worth knowing

- **One 3-class model, not two hazards.** A single `HistGradientBoostingClassifier`
  with target {0 survive, 1 default, 2 payoff} gives `predict_proba` columns (sorted
  class order) = `[1-h_d-h_p, h_d, h_p]` directly. This guarantees the hazards sum to
  ≤ 1 and avoids duplicating the person-period frame per cause.
- **Weekly person-period.** `terminal_week(days) = clip(ceil(days/7), 1, 13)`; each
  loan emits rows for weeks `1..T_i`. Weekly (not daily) keeps the frame ~440k rows
  and aligns exactly with B's 13-week grid (day `7a`). Payoff lands at week 9 (day 60);
  the day-90 rule maps to weeks 9–13.
- **`hazard_curves` builds 13 synthetic age-rows per applicant** (the same static `x`
  repeated with `loan_age_weeks = 1..13`) and reads per-week hazards off `predict_proba`.
  `loan_age_weeks` is a real model feature — that's what shares strength across ages.
- **B monotonicity is free.** `CIF_d(t)` is non-decreasing by construction, so per-cohort
  B curves are already monotone; `survival.enforce_monotone` (cummax) is just a guardrail.

## 5. Deliverable-specific gotchas

- **A decision uses the UPPER PD bound, not a threshold.** Expected NPV is strictly
  *decreasing* in PD (repaid return `g = 0.0875` of principal vs ~`-(1-rec)` on
  default), so approve iff `expected_npv(pd_upper, amount, rec) > 0`. Using the lower
  bound is backwards (it's the optimistic case). Break-even PD ≈ 8% (full-principal
  convention).
- **B is over APPROVED loans, conditioned on A's decisions.** Because we approve only
  low-PD loans (~19%), the approved-cohort cumulative default (cdr@age13 ≈ **5%**) is
  much lower than the historical funded-book rate (~20.6%) and is *not* 1.0. A useful
  internal cross-check: approved-cohort PD mean ≈ B cdr@age13 (~5%) — if they diverge,
  something is wired wrong.
- **C is `do()`, not a feature swap.** `dag.propagate_intervention` sets the feature
  AND its deterministic descendants: `has_linked_bank_feed` toggles the whole bank-feed
  block (+ missingness flags); `requested_amount` / `observed_monthly_revenue_avg_3mo`
  recompute the engineered ratio. Everything else is observational and is flagged as
  such — full SCM/DML is `[FUTURE]`, not claimed.
- **Counterfactual queries reference applicants in `decision_frame`** (val+test), not
  train. Look raw rows up by `applicant_id` there.

## 6. Calibration & uncertainty

- **The model mildly under-predicts** (mean predicted 0.185 vs actual 0.206 on the
  validation funded subset) because it's trained on the selected/funded slice. The
  upper-bound decision rule partly compensates. Isotonic recalibration of the
  *submitted* PDs is implemented but not yet applied to the final output (open TODO).
- **Intervals are ensemble percentile bands** (5/95 across `N_BAG_SEEDS=5` seeds), not
  conformal. Split-conformal is `[FUTURE]`. `calibration.clip_and_order` is the final
  safety net that enforces `lower ≤ point ≤ upper` and `[0,1]` before writing.

## 7. Pipeline assembly invariants

- Score the **`decision_frame`** (val+test), not train, for A.
- Map A's decisions back onto the decision-frame row order **by `applicant_id`**
  before building B (B needs both `cohort_week` via `data.assign_cohort_week` and the
  decisions array).
- Cast `decision`, `cohort_week`, `loan_age_weeks` to `int`; write with `index=False`.
- Fill-then-clip order matters: LEFT-JOIN → fill gaps → `clip_and_order` →
  assert no NaN.

## 8. Reproducibility & git

- **Unzipped `dataset/{train,validation,test}.csv` are gitignored** — reproducible via
  `unzip dataset/dataset-compressed.zip -d dataset/` (the zip *is* committed). Don't
  commit the CSVs.
- **`submission/*.csv` ARE committed** (versioned deliverables).
- This repo's git policy differs from the sibling repos: here Claude commits & pushes
  to the private `origin` at checkpoints (Hossain often lacks local access). After
  pushing, confirm `0 ahead / 0 behind`.

## 9. Multi-agent / workflow learnings

- **Build many interdependent files with: disjoint file ownership + one shared
  interface contract + structured-output schemas.** That's how `src/smb/*` was built
  without interface drift. Tier the fan-out by dependency (foundation → models →
  integrate).
- **Do not trust a workflow's self-report.** The integration agent here claimed to be
  waiting and left Deliverable **C unwritten** — A and B existed, C did not. Always
  re-run the pipeline and the official validator yourself before believing "PASS."
- **Adversarial verification caught a real error.** The correctness workflow
  (`docs/VERIFICATION.md`) flagged the prior draft's *lower*-PD-bound decision rule as
  mathematically backwards and corrected the break-even number — before it was written
  into the methodology. Recompute from data; refute, don't restate.

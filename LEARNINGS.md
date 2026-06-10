# Learnings

Non-obvious patterns, gotchas, and operational knowledge for this repo
(SMB Underwriting Challenge). Complements `CLAUDE.md` (preferences, commands) and
`METHODOLOGY.md` (the methodology of record) with the deeper insights that bit us
while building it. Every empirical number here was recomputed from the data
(`scripts/eda.py`); the technical claims were adversarially verified
(`docs/VERIFICATION.md`).

> **Platform note.** This was developed locally on Windows, but the repo is normally
> driven from **Claude Code on the web (Linux cloud)**, where the pipeline runs
> unchanged with plain `python`. Anything below tagged **[Windows-local only]** is a
> Windows-console workaround you can ignore on the web. The modeling logic is
> platform-agnostic; only the shell ergonomics differ.

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
  identified*. The IPW / reject-inference **reweighting path has been removed**;
  `src/smb/propensity.py` is now a **pure diagnostic** (`deterministic_funding_rule`
  *proves* the threshold; `out_of_support_fraction` + `positivity_report` quantify
  the overlap failure) and never touches the fit. The hazard model is fit
  **unweighted**. Do not "add IPW to improve it" — it's mathematically unsound here.
- **Exclude the `prior_*` selection columns from the outcome model.** `prior_decision`
  is constant (`==1`) in the labeled set (no variance, pure selection); 
  `prior_approved_amount` is funded-only leakage; and crucially
  `prior_underwriter_score` is the score the funding THRESHOLD is defined on — the
  labeled set only ever has `score >= 0.273`, yet **~44% of the decision population is
  below that minimum** (out of training support). All three live in
  `features.EXCLUDE_COLS`; the score is kept only for the funding-rule diagnostic. If
  you ever see `prior_underwriter_score` back in the model matrix, that's the bug.
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

## 3. Environment & stack

- **sklearn-only stack (platform-agnostic).** The flexible learner is
  `HistGradientBoostingClassifier` (native NaN handling + `sample_weight`); isotonic
  via `IsotonicRegression`; everything else hand-rolled. The code imports *only*
  sklearn, so it runs identically on the web/Linux runtime and on the local box. **Do
  not add** lightgbm/xgboost/lifelines/econml/doubleml — they're `[FUTURE]`, not a
  `pip install`. (Original reason: the local machine is Python 3.14, which has no
  wheels for those; keeping the import surface to sklearn means behavior is the same
  everywhere, including on the web.)
- **Keep model features numeric (no `category` dtype).** Integer-coded categoricals
  are treated as numeric/ordinal. Deliberate robustness call: pandas `category` dtype
  causes cross-split category-alignment bugs between train / decision-frame /
  counterfactual frames. `build_features` returns all-float64 with a stable, sorted
  column order; `align_columns` reindexes other frames to match.
- **[Windows-local only]** the local console is cp1252, so locally I run
  `PYTHONUTF8=1 python ...` and keep `print()` ASCII-only (unicode like `π`/`≈`/`≤`
  raises `UnicodeEncodeError`). **On the web/Linux this does not apply** — UTF-8 is
  default, use plain `python`. (Unicode in `.md` files is always fine; this was only
  ever about Windows *stdout*.)

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

- **The model under-predicts out-of-time, and we now correct it.** On the validation
  funded subset the train-fit ensemble predicts mean 0.187 vs realized 0.206 (the
  later cohorts default more — temporal drift on top of selection). We fit an
  **isotonic OOT calibrator** on that subset and **apply it to the submitted A/B/C
  PDs** (`config.APPLY_OOT_CALIBRATION`, in `pipeline.fit_oot_calibrator`). Two traps
  to know: (1) **never report isotonic's in-sample ECE** — it's ~0 by construction;
  gate on a **K-fold cross-fitted** held-out ECE instead (here 0.0228 → 0.0199). (2)
  The correction is small in ECE but **large in decisions**: because the model
  under-predicts *near break-even* (raw 0.06 → calibrated ≈ 0.096, vs PD\* ≈ 0.088),
  approvals tighten from ~21% to **~7%**. That's the honest consequence, not a bug —
  but it means the calibration is decision-critical, so apply the **same** monotone
  map to A and B (so `cdr@age13 ≈ approved-set PD` stays consistent) and remember the
  calibration set is itself the *funded* slice (a stated extrapolation caveat).
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
- **Git policy here is the OPPOSITE of the sibling repos.** The other repos say
  "output the commit command, don't run it." Here, because work is driven from the web
  (no local fallback), Claude **commits and pushes to the private `origin` itself** at
  every checkpoint, then confirms `0 ahead / 0 behind`. Unpushed commits at end of
  session = lost work. Don't carry over the sibling repos' hands-off git rule.
- **"Reproducible" is a three-part claim: (seed, budget, sklearn version).**
  HistGradientBoosting output shifts at the third decimal across sklearn releases —
  the identical c_proxy call gave gcomp 0.0854/naive 0.1085 under sklearn 1.8.0
  (this repo's Python, which built the committed artifacts) and 0.0869/0.1087 under
  1.9.0 (the harness venv). Two agents both honestly claimed "bit-for-bit
  reproduction" of *different* numbers because each reproduced in its own
  environment. Consequences: `requirements-dev.txt` pins `scikit-learn==1.8.0`;
  any quoted figure's provenance must name the environment, not just the config;
  and a "doesn't reproduce" finding means *check the environment first*, not
  "the artifact is stale."

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

## 10. Synthetic-world gotchas (from the sibling harness)

- **A synthetic world's "random selection" knob can silently share an exogenous
  draw with an OBSERVED column — check corr(selection noise, observed columns)
  before trusting a selection-severity sweep.** In the sibling harness
  (`closed-loop-default-detection`), the SCM's selection blend reused the
  exogenous draw behind the observed `prior_underwriter_score` column: corr
  ≈ 0.92 between the selection score and an observed feature at severity 0, and
  an in-sample propensity model hit AUC ≈ 1.0 on what was supposed to be
  selection-at-random. That inverts the severity semantics ("severity 0" is
  supposed to mean selection *no* propensity model can explain), and the
  inversion is **unfalsifiable from inside the world** — every downstream
  metric still computes, the sweep still runs, the numbers just mean the
  opposite of their labels. Caught by recon *before* implementation; fixed with
  a gated `independent_selection_noise` flag (default off, a dedicated frozen
  noise node drawn after all existing draws, so the default RNG stream is
  sha256-identical and the 51/51 fidelity gate stays green). General check: at
  severity 0, correlate the selection noise against every observed column and
  fit a propensity model — you want corr ≈ 0 and AUC ≈ 0.5, not 0.92 and 1.0.
- **Known + deliberately not fixed before the freeze:** the harness's
  `requested_amount_to_observed_revenue` is derived from *ungated* bank-feed
  revenue, so no-feed rows carry information that structural missingness says
  they shouldn't have (`scm.py:666-673`). Disclosed in the harness `FABLE.md`.
  Decision: fixing it alters the SCM, which invalidates every verified number
  (5-seed sweep, unified frontier, 51-check fidelity gate) and forces a full
  re-verification cycle — disproportionate for a realism nuance in the
  *validation* world that does not touch the real-data deliverables. Fix it
  first thing if the harness outlives the hackathon.

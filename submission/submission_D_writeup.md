# Deliverable D — Technical Writeup

**Team:** Closed-Loop Underwriting

> Numbers below are recomputed from the raw CSVs and from a proxy scorecard
> (`scripts/run_scorecard.py`) measured where ground truth exists. The TRUE
> competition score is not locally computable (hidden test labels + hidden
> per-term normalization), so every figure we quote is a **proxy** measured on
> the out-of-time **validation funded** holdout (2,551 loans applied after the
> train window), aggregated with the brief's published weights (p.14).

## 1. Problem framing & assumptions violated

This is a **selective-labels** underwriting problem, not a clean classification
task. Outcomes (`default_flag`, timing, recovery) exist only on funded+matured
loans — 51,722 of 85,340 train rows; **0 of 8,817 test rows**. Three standard
assumptions break, and each changed our design:

- **Positivity fails globally.** The legacy funding rule is *deterministic*:
  `prior_decision == 1 iff prior_underwriter_score ≥ ~0.273`, with zero mismatches
  and a clean gap (max declined 0.27297 < min approved 0.27301). There are no
  funded look-alikes for declined applicants, so IPW / doubly-robust estimators
  are **not identified**. We therefore fit the outcome model unweighted and treat
  declines by **bounding / abstention**, never silent extrapolation. We also
  **exclude `prior_underwriter_score` and its consequences** (`prior_decision`,
  `prior_approved_amount`) from the PD features to avoid encoding the selection.
- **The payoff is sharply asymmetric and time-dependent.** A repaid loan nets
  `0.0875·R` (3% fee + 35%·60/365 interest); a default destroys most of `R` net of
  recovery. Crucially, default *timing* is bimodal: in-term defaults span days
  3–60, then a spike at exactly day 90 (the observation-window close) with **zero**
  mass in (60, 90). The day-90 spike recovers ≈0 (empirical mean recovery fraction
  **0.0001** vs **0.118** in-term) — these are near-total losses, not near-payers.
- **Missingness is structural.** The six bank-feed columns are null *iff*
  `has_linked_bank_feed == False`. We preserve NaN (HistGB splits on it) and add
  explicit `*_was_null` indicators rather than imputing.

## 2. Methodology

One **weekly competing-risks discrete-time hazard** (default vs payoff) is the
spine for all four deliverables. A single 3-class `HistGradientBoostingClassifier`
on a person-period expansion yields per-week hazards `h_d, h_p`; the competing-risks
recursion gives `CIF_d(t)` and lifetime PD. Deliverables read off one fit. Stack
is **sklearn-only** (HistGB + isotonic); features are all-float with NaN preserved.
Every figure is proxy-measured on the out-of-time validation funded holdout
(`scripts/run_scorecard.py`); the four deliverables land as:

| Deliverable | Proxy metric (OOT funded holdout, n=2,551) | Result | Comparison |
|---|---|---|---|
| **A — policy** | realized portfolio NPV | **$603,817** (approve 59%) | flat $512,660 (27%); legacy/approve-all -$2.56M; oracle $4.75M |
| **A — calibration** | 90% PD-band coverage @ mean width | **0.89 @ 0.134** | raw ensemble 0.80 @ 0.082 |
| **B — trajectory** | cohort-weighted CDR MAE | **0.0150** | pre-recal 0.0207; naive train-marginal 0.0259 |
| **C — counterfactual** | g-comp vs naive MAE (synthetic harness, sev 0.4) | **0.0857 vs 0.0991** | gap +0.0134, positive on 24/25 seeds |

![Deliverable A: realized portfolio NPV on the 2,551-loan OOT funded holdout. The timing E[NPV] rule clears the flat break-even baseline by +$91K and turns the legacy book's -$2.56M into a profit, against a $4.75M hindsight ceiling.](reports/pnl_backtest.png)

- **Deliverable A — timing-integrated NPV policy.** We replaced a flat
  break-even rule (`approve if PD < ~8.9%`) with an **expected-NPV rule that
  integrates over default timing**: `E[NPV_i] = Σ_t P(default in week t)·NPV(t) +
  P(repay)·0.0875R`, where `NPV(t)` credits the daily ACH draws a *late* in-term
  defaulter pays before defaulting, and day-90-window defaults are charged as
  total losses (no term draws). Approve iff `E[NPV] > 0`. A flat PD threshold
  over-declines profitable late-defaulters; the timing rule recovers them.
  **Realized backtest (high compute):** timing earns **$603,817** (approving 59%)
  vs the conservative flat rule's **$512,660** (approving 27%) on the 2,551-loan
  funded holdout (**+$91,157**), against −$2.56M for approve-all/legacy and a
  $4.75M hindsight ceiling. The extra approvals timing wins back are exactly the
  late-in-term defaulters whose pre-default ACH draws keep their E[NPV] positive.
- **Deliverable B — cohort trajectory.** Approved-cohort `CIF_d` averaged per
  (cohort_week, age), monotone by construction. The hazard mildly under-predicts
  out-of-time (lifetime ratio ≈1.12), so we apply a **single-parameter OOT
  recalibration** fit on the validation funded subset: weighted CDR MAE drops from
  **0.0207 → 0.0150** (naive train-marginal baseline 0.0259).
- **Deliverable C — counterfactuals.** `do(feature=value)` via the causal-graph
  surgical intervention (see §3).

All scaling is governed by one `--compute {low,med,high}` flag (ensemble size,
conformal splits, harness grid) and is deterministic per (seed, budget).

## 3. Causal reasoning & counterfactual methodology

Deliverable C asks for an **interventional** `P(default | do(X=x))`, not the
observational `P(default | X=x)`. Our submitted estimator applies `do(X=x)` to the
applicant's raw row, **propagates the deterministic descendants** via the DAG
(`dag.propagate_intervention`: engineered ratios recomputed; toggling
`has_linked_bank_feed` rewrites the whole bank-feed block + its missingness
flags), then reads lifetime PD off the hazard ensemble. For features with
SCM descendants this is **standardization / g-computation in spirit**; for the
rest it is a disclosed observational perturbation — we say so plainly rather than
overclaiming a full SCM on real data where positivity fails.

Because real data **cannot** validate interventional accuracy (we never observe
counterfactuals), we built a **fidelity-gated synthetic validation harness**
(`closed-loop-default-detection`): a structural causal model tuned to match the
real marginals (fidelity gate green), with a true `do()` oracle. There we grade a
deployable **g-computation estimator** (fit on approved rows only, no SCM
coefficients) against **naive conditioning**. Certified across a **25-seed sweep**
(900 queries each): on the **strong-propagation** slice at moderate selection
severity (0.4), g-computation MAE is **0.0857 ± 0.0151 vs naive 0.0991 ± 0.0190**
— gap **+0.0134 ± 0.0085, positive on 24/25 seeds** (≈8 standard errors above
zero; the one flip, −0.004, is reported, not buried), a **~13% relative
reduction** in interventional error exactly where naive conditioning is
structurally wrong (it ignores that intervening on a parent moves its children).
A 5-seed pilot gave the same mean with a narrower interval — scaling confirmed
the mean and exposed the variance the small sweep understated.

| Selection severity | naive MAE | g-comp MAE | gap (mean ± sd) | seeds gap > 0 |
|---|---|---|---|---|
| **0.4** (operating frontier) | 0.0991 | 0.0857 | **+0.0134 ± 0.0085** | **24 / 25** |
| **1.0** (full unobs. confounding) | 0.1026 | 0.1009 | +0.0017 ± 0.0020 | 20 / 25 |

The harness measures where this stops working as a **collapse curve**, not an
asserted cliff: sweeping severity {0.4, 0.6, 0.8, 1.0} (paired seeds) the gap
falls **+0.0133 → +0.0059 → +0.0050 → +0.0017**, most of it across the 0.4 → 0.6
boundary where the IPW frontier also breaks (§4, fig. below), then a noise-flat
plateau, then the floor. At **full severity** the advantage is negligible
(+0.0017 ± 0.0020, sign-flipping on 5/25 seeds) and we claim none. One structural
mechanism bounds both fixes: estimators fit on approved rows whose conditionals
are distorted by selection on an *unobserved* confounder — backdoor adjustment
cannot fix unobserved confounding, IPW cannot fix broken positivity — measured
independently in the same synthetic world. **Regulator defense:** drivers are
reported as interventional effects with the propagation made explicit and
non-intervenable identity features (sector, vintage) refused — not raw
observational correlations — and the validation states both halves: the direction
of the method is certified with error bars across seeds, *and* the regime where
it stops working is measured, not assumed away.

## 4. Calibration & uncertainty quantification

The scoreboard rewards 90% intervals that **cover ≈0.90 at minimum width**, and
decisions flip in a narrow band near break-even, so calibration is first-class.

- **A (PD).** Raw ensemble percentile bands measure model *disagreement*, not
  predictive uncertainty for a binary outcome, and they **under-cover** (binned
  coverage 0.80 at high compute). We use a **split-conformal** half-width: bin by
  predicted PD, take the 0.95-quantile of per-bin `|empirical_rate − predicted|`,
  fit on the validation funded subset. Conformity is measured on the **raw** point —
  a *fitted* recalibrator (isotonic) shrinks in-fold error and under-covers
  out-of-fold (we verified raw → 0.875 vs isotonic → 0.53 held-out coverage). Result
  on the holdout (evaluated by repeated 50/50 split-conformal, no circularity):
  **coverage 0.89 at width 0.134** (vs raw 0.80 / 0.082).
- **B (CDR).** Ensemble percentile bands across members, monotone per cohort.
- **C (PD_cf).** Ensemble percentile bands across members; population fallback for
  unscoreable queries so the file is never null.

**Declined-subpopulation coverage** is the part real data can never check; the
harness measures it against SCM truth — and the selective-labels loop now runs
**on the SCM itself**, the same synthetic world as the §3 counterfactual results.
IPW holds declined-cohort ECE at **0.036 / 0.038 / 0.097** for severity
0 / 0.2 / 0.4 (pass, seed 42), then fails at 0.6 (**0.244**) — an honest operating
frontier at severity 0.4, the same boundary where the g-computation advantage
collapses (§3), measured in one world rather than two.

![Operating frontier in the closed-loop-default-detection harness: PD calibration error (ECE) on the declined applicants real data can never score, vs selection severity. IPW (the operating estimator) holds below the 0.10 target through severity 0.4, then all three estimators break together at 0.6 — the same boundary at which the §3 g-computation advantage collapses.](reports/clue_frontier_scm.png)

## 5. Limitations & what we'd do differently

- **Proxy ≠ truth.** All gains are proxy-measured on one OOT holdout; the hidden
  normalization could reweight terms. We mitigated by recomputing every number
  from raw CSVs and cross-fitting calibration to avoid optimism.
- **The day-90 spike convention is load-bearing** for P&L (30%). We charge it as a
  total loss (no term draws) on strong empirical grounds (recovery ≈0.0001), but a
  different draw-accounting convention would move the P&L level.
- **C is g-computation-in-spirit, not a fitted SCM on real data** — positivity
  failure makes a full real-data SCM unidentified. The harness certifies direction
  across 25 seeds (24/25 positive) **only inside the severity ≤ 0.4 frontier**; at
  full selection severity the advantage collapses by nearly an order of magnitude
  (+0.0017 ± 0.0020, sign flips on 5/25 seeds), and the MAE gain trades a small but
  consistent increase in negative bias (g-comp bias more negative than naive on
  25/25 seeds) — measured limits, not the exact real-data magnitudes.
- **Three "obvious next steps" are tested negatives, not future work.**
  (a) *Conformal-into-the-decision:* re-pricing E[NPV] at the conformal upper PD
  bound (`scripts/exp_conformal_decision.py`; out-of-fold, half-width fit on one
  half of the funded holdout, realized P&L of the decisions on the disjoint
  half, 10 splits × 2 orderings) cuts approval 0.60 → 0.19 and loses
  **$137K ± $182K** per test half vs the timing rule (wins 7/20 splits) — a
  coverage-calibrated *interval width* is not a decision-optimal shift.
  (b) *Per-cohort OOT recalibration for B:* with empirical-Bayes shrinkage, the
  method-of-moments between-cohort variance on the full holdout is exactly
  **zero** (cohort spread fully explained by binomial noise at ~200
  loans/cohort), so the estimator degenerates to the global factor; out-of-fold
  (`scripts/eval_b_recal_oof.py`, 50 paired evals) per-cohort factors never win.
  (c) *Signed OOT level correction in the decision:* scaling A's lifetime PD by
  the measured OOT factor (the B-recal ratio ≈1.107, fit on the cal half) before
  the E[NPV] integration (`scripts/exp_oot_decision.py`, same OOF protocol). This
  is the *right shape* the symmetric band (a) lacked — a measured shift, not an
  interval width, and it behaves as designed (approval 0.60 → 0.55) — but it
  still does not improve realized P&L: **−$4.8K ± $57K** per test half, wins
  10/20 (a coin flip). The marginal loans the up-shift newly declines were, on
  net, profitable on this holdout. All three code paths ship dormant behind
  flags, with the experiments committed.
- **With another day:** bootstrap bands for C scaled by compute. (We also
  explored five g-computation estimator variants in the harness — richer
  ancestor parent-sets, bagged child mechanisms, stronger regressors, post-arm
  debiasing, fixpoint propagation — but none improved the strong-propagation MAE
  across all five seeds, so the deployed estimator is at its achievable frontier
  on that slice; see `docs/VERIFICATION.md`.)

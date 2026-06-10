# Methodology & Writeup Plan — SMB Underwriting Challenge

This document is our methodology of record. It is written to be **judge-ready and
honest**: it separates what this repository *actually does today* from what we
*propose* as the modeling approach, and it states the identification limits of the
problem rather than papering over them. Every empirical number below was recomputed
from the released CSVs (see `scripts/eda.py`), and the technical claims were
adversarially cross-checked before being written here (see
[`docs/VERIFICATION.md`](docs/VERIFICATION.md) for the method and findings).

The strongest contribution of this submission is **not** a heavyweight model — it
is a disciplined account of *what is identifiable, what is only partially
identifiable, and what is pure extrapolation* in this dataset, and a baseline that
acts accordingly.

## Status legend

Every method below is tagged with its honest status:

- **[VALIDATED]** — implemented and runs in this repo today (exploration / guardrails).
- **[BASELINE]** — proposed; feasible now with the available stack (numpy/pandas/
  scikit-learn); what we would submit.
- **[STRETCH]** — proposed; a reach within the hackathon window.
- **[FUTURE]** — research-grade extension; **not** implemented and not claimed as such.

---

## 0. What this repository is

This repo is primarily an **observability / exploration layer plus a validated
scaffold**. Concretely, it **[VALIDATED]**:

- verifies the dataset's structural constraints (split shapes, the 13,306-applicant
  decision set, the 13-week cohort partition);
- quantifies the **selective-labels** structure and the legacy funding rule;
- characterizes **missingness** structure (which blocks are null and why);
- measures **default timing** and the recovery distribution;
- describes the **intervention-query** structure for Deliverable C;
- runs **selection / overlap diagnostics** (positivity);
- wraps the **official validator** so any produced submission is gated.

On top of that layer, the **baseline tier (§2.1) is now implemented end-to-end**:
`python scripts/run_all.py` fits the weekly competing-risks hazard ensemble and
emits A/B/C that **pass the official validator**. What is still *not* implemented —
and is **not** claimed as done — is the stretch/aspirational machinery: IPW *folded
into the fit*, doubly-robust off-policy evaluation (DR-OPE), Double ML (DML), causal
forests, a Markov missed-draw simulation, split-conformal calibration, and DeepHit.
Those remain **proposed** tiers. (See §10 for the exact implemented-vs-proposed
inventory.)

---

## 1. Problem framing & assumptions violated

The lender maximizes realized portfolio value. With no binding capital constraint
the objective is separable across applicants, so the optimal policy is pointwise on
the sign of expected NPV. The payoff is **sharply asymmetric**: a repaid loan
returns only single-digit percent of principal, while an early default with low
recovery destroys most of it. So the task is to estimate a **calibrated expected
NPV**, not to maximize classification accuracy.

Four data realities — all verified — break standard assumptions:

**(a) Selective labels with a *deterministic* funding rule.** Repayment outcomes
exist only for funded, matured loans (60.6% of train; `prior_decision == 1`).
Critically, the legacy funding decision is a **perfect deterministic threshold on
an observed score**: `prior_decision == 1 iff prior_underwriter_score >= ~0.273`,
with **zero** mismatches across all 85,340 rows and a clean gap (max declined score
0.27297 < min approved score 0.27301). This means the funding propensity
`e(x) = P(funded | x)` is **degenerate (0 or 1)**: there is *no overlap*. The
practical consequence is severe and is the spine of our honesty story —
**positivity fails globally**, so IPW and doubly-robust estimators are **not
identified** for the declined region; the default behavior of declined applicants
is only reachable by *extrapolation*, never by reweighting funded look-alikes (there
are none at the same score). We therefore **fit the outcome model unweighted on the
funded labeled data only** and have **removed the IPW / reject-inference reweighting
path** entirely (`propensity.py` is now a pure diagnostic — it *proves* the rule and
reports positivity, but never reweights the fit).

A direct corollary drives a feature choice: because every funded (labeled) row has
`prior_underwriter_score >= 0.273` while **43.5% of the decision population sits
below that minimum** (recomputed from the CSVs), the score is *out of the outcome
model's training support for nearly half the applicants we must decide on*. Using it
as an outcome feature would extrapolate a legacy-policy artifact across the funding
boundary and bake selection into the default model. We therefore **exclude
`prior_underwriter_score` from the outcome model** (`features.EXCLUDE_COLS`) and keep
it **only** for the funding-rule / positivity diagnostic
(`propensity.deterministic_funding_rule`).

**(b) Out-of-time deployment.** Training spans 2024-01 to 2025-06; the decision
population (validation + test, 13,306 applicants) is a single later 13-week window
(2025-06-30 to 2025-09-28). We evaluate **out-of-time** on the validation split,
mirroring deployment — not a random split. Note validation labels also exist only
for its **funded** subset (2,551 / 4,489), so calibration on validation is itself
conditioned on the legacy policy.

**(c) Default is path-dependent and spread in time, with a point mass at day 90.**
Default triggers on 3 consecutive missed draws, 6 cumulative missed draws, or a
nonzero balance at day 90. Verified timing: all paid-in-full loans repay at
**exactly day 60**; defaults occur over days 3–60 **and then jump to a point mass
at exactly day 90** (22.5% of defaults), with **zero** defaults in the open
interval (60, 90). A single life-of-loan binary flag discards the timing that both
the NPV and Deliverable B require.

**(d) Missingness is informative (MNAR by design).** The bank-feed block is null
*iff* `has_linked_bank_feed == False` (verified exactly across all 98,646 rows);
`days_since_*` nulls mean "no prior event." We preserve missingness as signal
(indicator flags; native NaN handling) rather than blind-imputing. `prior_decision`
is constant (`== 1`) within the labeled set and is therefore excluded from the
outcome model, alongside `prior_approved_amount` (funded-only leakage) and
`prior_underwriter_score` (the legacy score the funding threshold is defined on; out
of support for ~44% of the decision set, see §1a). All three are informative only
for the funding/diagnostic model, not the outcome model.

---

## 2. Methodology, organized by feasibility tier

We deliberately tier the approach so the baseline is fully defensible on its own and
each higher tier is clearly labeled as proposed.

### 2.1 Baseline — what we would submit **[BASELINE]**

1. **Calibrated binary PD model.** A gradient-boosted classifier
   (`sklearn.ensemble.HistGradientBoostingClassifier`, which has native NaN handling)
   for lifetime default, with **isotonic recalibration** on a held-out (out-of-time)
   fold. Calibration — not ranking — is the scoreboard, because decisions flip in a
   narrow band around the break-even PD.
2. **NPV-based approval rule** (see §3) using the loan economics, with the
   **conservative upper-bound** decision criterion.
3. **Empirical cumulative-default-rate (CDR) trajectories** for Deliverable B,
   built from observed default timing and **monotone by construction** (see §5).
4. **Bagged / ensemble uncertainty intervals**: percentile bands from a seed/
   bootstrap ensemble of the PD model, used for both the decision criterion and the
   reported 90% intervals.
5. **Structural feature propagation** for Deliverable C where the dependency is
   deterministic (bank-feed block gating; engineered ratios), with all other
   counterfactuals reported as **observational** and flagged as such (see §4).

### 2.2 Stretch — competing-risks timing model **[STRETCH]**

A **cause-specific discrete-time competing-risks hazard** for the two absorbing
events, **default** vs **payoff**, fit on funded+matured loans in person-period
layout. Survival `S(t) = prod_{s<=t}(1 - h_d(s) - h_p(s))` and cumulative incidence
`CIF_d(t) = sum_{s<=t} h_d(s) S(s-1)`; lifetime `PD = CIF_d(H)`. Carrying the payoff
hazard prevents overstating PD (a loan that pays off is removed from default risk).
This would let A and B read off a single coherent fit. Given the verified timing
(payoff fixed at day 60; defaults as a [3,60] body plus a day-90 spike), even a
*weekly* discrete-time hazard is tractable and aligns with B's 13-week grid.

### 2.3 Aspirational / future work **[FUTURE]**

Explicitly *not implemented*, listed as the upgrade path:
doubly-robust off-policy evaluation (DR-OPE) and Double ML for the policy value;
causal forests for heterogeneous interventional effects; **conformal** prediction
for finite-sample interval coverage; a **Markov missed-draw simulation** that
reproduces the exact default automaton (3-consecutive / 6-cumulative / day-90
balance) from a latent daily-draw process; and DeepHit / Dynamic-DeepHit for
neural competing-risks. Each is a defensible direction; none is claimed as done.

---

## 3. Decision rule and loan economics

**Economics (from `dataset/README.md`).** Amount `R` in $5K–50K; 60-day term via
daily ACH draws; 35% APR; 3% origination fee; 90-day default window.

- Origination fee: `F = 0.03 R` (exact).
- Interest convention — **disclosed, not assumed away.** Reading "35% APR" as
  *interest on full principal for the full 60-day term* gives
  `R * 0.35 * 60/365 ≈ 0.0575 R`, hence a repaid-loan return of
  `F + interest ≈ 0.0875 R`. We treat **0.0875R as the full-principal upper bound**.
  Because repayment is via daily ACH draws (declining balance), a literal
  amortizing reading gives roughly half the interest (`≈ 0.029 R`, total return
  `≈ 0.059 R`). **The writeup will state the chosen convention explicitly**, since
  the headline break-even PD swings materially between them (below).
- Recovery: empirically `final_recovered_amount / R` on defaulters has mean
  ≈ 0.091, median ≈ 0.072, with ≈ 23% recovering exactly 0 (dollar-weighted ≈ 0.10).
  We use a central ≈ 0.09R prior and **disclose the dispersion**.

**Break-even PD.** With gain `g R` on repayment and loss `≈ (1 - rec) R` on default,
`NPV(PD) = gR - PD * R * (g + 1 - rec)` is **linear and strictly decreasing in PD**.
Setting it to zero, `PD* = g / (g + 1 - rec)`:

- Full-principal convention, zero recovery (true worst case): **PD\* ≈ 8.05%**.
- Full-principal convention, empirical recovery ≈ 0.10R: **PD\* ≈ 8.7–8.9%**.
- Amortizing convention: **PD\* ≈ 5.5–6.1%**.

So we report a break-even **band** (≈ 8.0–8.9% under the full-principal convention)
rather than a single spurious "8.3%". The funded-population default rate (17.45%) is
well above any of these, confirming the legacy book funds many negative-NPV loans —
the headroom our policy targets.

**Conservative decision rule (corrected).** Because NPV is **decreasing** in PD, the
pessimistic case for any applicant is its **highest** credible default probability.
We therefore:

> **Approve a loan only if its expected NPV remains positive when evaluated at the
> UPPER confidence bound of the predicted PD.**

This stress-tests each loan against its most pessimistic credible default rate: a
wide uncertainty band pushes the upper-bound PD high enough to make the loan
unprofitable, so **uncertain loans correctly default to denial**. (A prior draft
used the *lower* PD bound and is mathematically backwards — the lower bound is the
*optimistic* case, maximizes NPV, and would make ignorance default to *approval*.)

**NPV must separate the two horizons.** Interest accrual / cash flow is bounded by
the **60-day** contractual term, but default is observed through **day 90**. The NPV
expression accrues revenue over `t <= 60` and applies the default-loss term at the
observed default time (a [3,60] body plus the day-90 balance-rule spike), rather
than conflating the two windows.

---

## 4. Causal reasoning & counterfactual methodology (Deliverable C)

Deliverable C asks for an **interventional** quantity,
`P(default | do(f = v), X_{-f} = x_{-f})`, which differs from the **observational**
`P(default | f = v, ...)` whenever `f` is confounded. Re-scoring a predictor with one
feature swapped (and, equivalently, reading a SHAP attribution) answers the
*observational* question and is **not** a `do`-operator estimate. We say so plainly.

**What the baseline does honestly [BASELINE]:**

- **Structural propagation where the dependency is deterministic.** Interventions on
  `has_linked_bank_feed` propagate to the entire bank-feed block (verified strict
  gating), not a lone boolean; interventions on `requested_amount` /
  `observed_monthly_revenue_avg_3mo` propagate to the engineered ratio
  `requested_amount_to_observed_revenue`. These are mechanical, defensible edits.
- **Everything else is reported as observational** model response under feature
  substitution, with the **observational-vs-interventional gap surfaced**, not hidden.
- Query structure (verified): 900 queries / 300 applicants (3 each) / 30 features;
  ≈ 19% of queries (≈ 47% of distinct targeted features) hit features flagged
  `intervenable = False`. We distinguish **hard-structural identity** features
  (sector, geography_region, vintage_years, employee_count_bucket) from
  **merely-not-actionable history** features (account_age_days, prior_loans_count).
- The `has_linked_bank_feed` confounding story is made via **covariate association**
  (e.g., linkers vs non-linkers differ in aggregate_credit_utilization ≈ 0.40 vs
  0.47, requested_amount, and underwriter score) — **not** via raw default rates,
  which are nearly identical (0.207 vs 0.206) and would understate confounding.

**Limitation, stated as such [FUTURE].** A full structural-causal-model /
backdoor-adjusted / DML estimator is *not* implemented. Given the global positivity
violation (§1a), several interventions are only **partially identified**; we widen
intervals far from support and, where appropriate, abstain rather than report a
point effect as if it were a solved causal quantity. The `closed-loop-default-
detection` harness now demonstrates this exact limit empirically in its synthetic
world: at full selection severity, backdoor adjustment cannot repair selection on an
*unobserved* confounder — g-computation's advantage over naive conditioning collapses
to statistically zero (§9, WS4) — which is independent of, and does not soften, the
real-data positivity caveat above.

**Regulatory defense.** For each stated driver we would argue sign, magnitude, and
mechanism, and explicitly flag proxies — emphasizing that the model's drivers are
*associational* unless backed by the structural-propagation logic above.

---

## 5. Deliverable B — cohort default trajectories

For origination cohort week `w` (1–13) and loan age `a` weeks (1–13), B asks for the
cumulative fraction of **all approved cohort-`w` loans** that have defaulted by day
`7a`. Two correctness points the writeup makes precise:

1. **Denominator is all approved cohort loans**, not just defaulters. Hence
   `CDR_{w, a=13}` approximates the **approved-cohort lifetime default rate** — a
   **selection-dependent** quantity tied to *our* approval policy. It is **not** 1.0,
   and it is **not** the historical funded-book rate (≈ 0.206); it depends on whom
   *we* choose to approve.
2. **A defaulter-normalized timing curve is a different object.** The fraction of
   *eventual defaulters* that have defaulted by day `t` reaches 1.0 by day 90.
   Lifetime PD requires **scaling that timing shape by the cohort default rate**:
   `CDR(a) = cohort_default_rate * defaulter_normalized_timing(a)` (verified to
   machine precision). Conflating the two would massively overstate PD.

**Monotonicity by construction.** The validator requires `cumulative_default_rate`
non-decreasing in age within each cohort (tolerance 1e-9). Cumulative incidence is
non-decreasing by definition; as a guardrail the repo already provides
`survival.enforce_monotone` (a cummax) **[VALIDATED]**, applied before writing.

---

## 6. Calibration & uncertainty quantification

- **Calibration is primary, and now applied out-of-time. [BASELINE]** Decisions flip
  in a narrow band near PD\* ≈ 8.8%, so calibration — not ranking — is the scoreboard.
  We fit an **isotonic** calibrator on the **validation funded subset** (`n = 2,551`;
  its realized lifetime default, scored by the train-fit ensemble — a genuinely
  out-of-time check) and **apply it to the submitted A/B/C PDs** (`config.APPLY_OOT_
  CALIBRATION`). The map is monotone, so it preserves the `lower ≤ point ≤ upper`
  ordering and B's per-cohort age-monotonicity, and B is calibrated with the *same*
  map as A so `cdr@age13 ≈ approved-set PD` stays consistent. We gate the application
  on a **K-fold cross-fitted** ECE (held-out, not the isotonic's degenerate in-sample
  ~0): out-of-time ECE improves `0.0228 → 0.0199` and the mean predicted PD rises from
  `0.187 → 0.206` to match the realized `0.206`. The decision-relevant effect is large
  precisely because the model under-predicts *near break-even* (raw 0.06 → calibrated
  ≈ 0.096), which tightens approvals from ~21% (uncalibrated) to **~7%** — the correct
  conservative consequence of honest calibration. **Caveat (stated, not hidden):** the
  calibration set is the *funded* slice (positivity fails for the declined region), so
  applying the map across the whole decision population rests on a **smoothness /
  extrapolation assumption**, not identification; its effect concentrates in the
  in-support, near-break-even region where it is most trustworthy.
- **Intervals.** 90% bands from a bagged/seed ensemble (5th/95th percentiles) for A,
  B, and C. **[BASELINE]** Split-**conformal** coverage guarantees are the principled
  upgrade. **[FUTURE]**
- **Coverage checks** on the funded validation subset, with the caveat that this
  subset is itself selected. The repo already provides `calibration.coverage` and
  `calibration.clip_and_order` guardrails **[VALIDATED]**.
- **Off-policy value.** A doubly-robust estimate of our policy vs the logged policy
  would be the evaluation of record — but under the deterministic legacy rule
  (§1a) DR/IPW are valid **only** in common support, which here is empty across the
  threshold. We therefore treat policy value as **partially identified** and report
  bounds / abstention rather than a single DR number. **[FUTURE]**

---

## 7. Selective labels & overlap — treated honestly

- Labels exist only for funded+matured loans; declined and test outcomes are blank.
- The legacy policy is a **deterministic threshold** on `prior_underwriter_score`
  (§1a) → propensity is degenerate → **positivity fails globally**.
- IPW and doubly-robust estimates are valid **only** in common-support regions;
  here that region is essentially empty across the funding threshold.
- **Outside support we do not extrapolate silently.** We use wider uncertainty,
  partial-identification (Manski-style) language, or abstention. Any cross-threshold
  default prediction for declined applicants is an explicit **modeling extrapolation**
  (the default model as a smooth function of features across the boundary), labeled
  as such — its validity rests on a stated smoothness assumption, not on reweighting.

---

## 8. Limitations & what we'd do differently

- **Identification, not estimation, is the binding constraint.** The deterministic
  funding rule means no amount of reweighting recovers declined-applicant outcomes;
  honest bounds beat a confident-looking IPW number.
- **Interest convention is a genuine modeling choice** (full-principal vs
  amortizing) that moves the headline break-even from ≈ 8% to ≈ 5.5–6.1%. We pick
  one and disclose it.
- **Recovery is observed only on a selected sliver** of defaulters; we shrink to a
  prior and disclose dispersion.
- **The competing-risks timing model is the highest-value upgrade** and is only at
  the stretch tier here.
- **C answers are observational** outside the structural-propagation cases; we report
  the gap rather than claim solved causal effects.

---

## 9. Related work

- **Reject inference / selective labels.** Kozodoi, Lessmann, Alamgir,
  Moreira-Matias & Papakonstantinou (2025), *Fighting Sampling Bias: A Framework for
  Training and Evaluating Credit Scoring Models*, EJOR 324(2):616–628; Lessmann,
  Baesens, Seow & Thomas (2015), EJOR 247(1):124–136; and the "selective labels"
  framing of Lakkaraju, Kleinberg, Leskovec, Ludwig & Mullainathan (2017), KDD.
- **Profit-based credit scoring (Expected Maximum Profit).** Verbraken, Verbeke &
  Baesens (2013), *A Novel Profit Maximizing Metric…*, IEEE TKDE 25(5):961–973; and,
  applied to credit, Verbraken, Bravo, Weber & Baesens (2014), *Development and
  application of consumer credit scoring models using profit-based classification
  measures*, EJOR 238(2):505–513.
- **Discrete-time survival for lifetime PD.** Bellotti & Crook (2013), *Forecasting
  and stress testing credit card default using dynamic models*, IJF 29(4):563–574;
  Bellotti & Crook (2009), JORS 60(12):1699–1707. (The IFRS 9 ECL discrete-time
  survival framing is post-2018 and cited separately, not attributed to Bellotti &
  Crook.)
- **Conformal prediction.** Angelopoulos & Bates (2023), *Conformal Prediction: A
  Gentle Introduction*, FnT in ML 16(4):494–591 (arXiv:2107.07511).
- **Policy learning / heterogeneous effects (and their caveats).** Wager & Athey
  (2018), *Estimation and Inference of Heterogeneous Treatment Effects using Random
  Forests*, JASA 113(523):1228–1242; Athey, Tibshirani & Wager (2019), *Generalized
  Random Forests*, Annals of Statistics 47(2):1148–1178; Athey & Wager (2021),
  *Policy Learning With Observational Data*, Econometrica 89(1):133–161.
- **"Attribution is not intervention."** Lundberg & Lee (2017), *A Unified Approach
  to Interpreting Model Predictions*, NeurIPS — with the explicit non-causal caveat
  from the SHAP documentation; Pearl (2009), *Causality* (2nd ed.), Cambridge UP
  (do-calculus: Pearl 1995, Biometrika 82(4):669–688).

---

## 10. Implementation status (implemented vs proposed)

**Implemented today — runs, validator-passing [VALIDATED] + [BASELINE]:**

- `scripts/eda.py` — exploration/validation pass (split shapes; 13,306 decision set;
  selective-labels gap; deterministic funding-threshold check; default & payoff
  timing; recovery distribution; bank-feed gating; intervention-query structure;
  overlap diagnostics; cohort sizes).
- The **full baseline modeling pipeline** (`data`, `features`, `survival_data`,
  `survival`, `recovery`, `policy`, `dag`, `causal`, `calibration`, `pipeline`):
  weekly **competing-risks hazard ensemble** (3-class HistGradientBoosting, 5 seeds)
  → lifetime PD → **out-of-time isotonic calibration** (§6) → **upper-bound NPV
  decision** (A), cohort CIF trajectory (B), and `do()` counterfactuals with
  structural propagation (C). The outcome model is fit **unweighted on funded labeled
  data only**, **excludes** `prior_underwriter_score` (§1a/d), and the IPW /
  reject-inference reweighting path has been **removed**; `propensity.py` now only
  *proves* the deterministic funding rule and reports positivity.
- `scripts/run_all.py` fits the ensemble and writes A/B/C; the official
  `validate_submission.py` returns **`RESULT: PASS`** (0 errors). Sanity (current):
  **~7% approval** (down from ~21% uncalibrated — honest out-of-time calibration
  reveals the model under-predicts near break-even); approved-set PD ≈ 0.041 (every
  approval has upper-bound PD < 0.0838 < break-even 0.0879) vs ≈ 0.28 declined; B
  monotone with cdr@age13 ≈ 0.047 (≈ approved-set PD, cross-check holds); out-of-time
  calibration on the validation funded subset (mean predicted **0.187 → 0.206** to
  match actual **0.206**; cross-fit ECE **0.0228 → 0.0199**).
- Stack is **sklearn-only** (HistGradientBoosting + IsotonicRegression); no
  lightgbm/xgboost/lifelines/econml.

**Proposed / NOT implemented (do not claim as built):**

- The **broader model family** — LightGBM / XGBoost / a logistic stacker and richer
  diversity-driven bagging — is a **[FUTURE]** extension, *not* a pip-install here:
  the shipped stack is deliberately sklearn-only so behaviour is identical on every
  Python ≥ 3.11 runtime (the original box is Python 3.14, which lacks wheels for
  them). The hazard spine is model-agnostic, so swapping in those learners is a clean
  upgrade path.
- IPW / reject-inference reweighting (now **removed**, not merely unused — positivity
  fails, §1a), DR-OPE, Double ML, causal forests, **split-conformal** intervals (we
  ship ensemble percentile bands), a Markov missed-draw simulation, and DeepHit /
  Dynamic-DeepHit.

**The submission must keep this implemented-vs-proposed boundary visible.** Use "we
validate," "we use as our baseline" (for the above), "we propose," and "future
extension" (for the list just above) — never claim the research stack is built.

---

## 11. Open TODOs

1. **[done]** Interest convention chosen and stated: full-principal (`g = 0.0875`),
   used in `policy.expected_npv`.
2. **[done]** Recovery: empirical mean `≈ 0.091R` (`recovery.estimate_recovery_rate`),
   dispersion disclosed in §3.
3. **[done]** Baseline tier implemented end-to-end; `run_all.py` → `RESULT: PASS`.
4. Distill §1–§8 into the 4-page Deliverable D writeup
   (`submission_D_writeup_template.md` → PDF), leading §3 (causal) with the
   identification story.
5. **[done]** Methodology-of-record outcome model: the legacy
   `prior_underwriter_score` is **excluded** from the outcome model and the IPW /
   reject-inference path is **removed** (§1a — positivity fails under the
   deterministic funding rule). Out-of-time recalibration of the *submitted* PDs is
   carried by the **WS2/WS3** estimators in §9 (B: `survival.fit_cif_scale`;
   A: split-conformal band), which the proxy scorecard measures against. The
   isotonic `oot_iso` path (`config.APPLY_OOT_CALIBRATION`,
   `survival.predict_trajectory(oot_iso=)`) remains available as an alternative
   recalibrator but is **not** the shipped one after the WS1–WS5 merge.
6. Add the stretch/aspirational tiers in §2.2–§2.3 (and the [FUTURE] model family —
   LightGBM/XGBoost/logistic stacking) as time allows.

---

## 9. Upgrade addendum — score-weighted optimization (WS1–WS5) [VALIDATED via proxy]

A proxy scorecard (`scripts/run_scorecard.py` → `reports/scorecard.json`) now
optimizes against the brief's published weights, measured where ground truth
exists (train funded+matured for the fit; the **out-of-time validation funded**
subset, 2,551 loans, as held-out eval). The TRUE score is not locally computable
(hidden labels + hidden normalization) — every number below is a documented proxy.
**Step-0 baseline → final (compute=high): weighted_proxy 0.3499 → 0.5583.**

- **WS1 — A decision rule is now TIMING-INTEGRATED, not flat-at-upper-PD.**
  `policy.portfolio_decisions(rule="timing")` approves iff `E[NPV] > 0` where
  `E[NPV] = Σ_t P(default week t)·NPV(t) + P(repay)·0.0875R`
  (`economics.expected_npv_timing` over `survival.default_week_probs`). It credits
  the daily draws a *late in-term* defaulter pays; day-90-window defaults are
  total losses (empirical recovery ≈0.0001 vs 0.118 in-term, two rates from
  `recovery.estimate_recovery_rates_by_timing`). Realized OOT backtest: timing
  **$570,299** vs flat **$355,786** (+$214,512); the flat rule over-declines
  profitable late-defaulters. `config.POLICY_RULE`/`POLICY_CONSERVATIVE` select it.
- **WS2 — B trajectory gets a single-parameter OOT recalibration** (the hazard
  under-predicts, lifetime ratio ≈1.12): `survival.fit_cif_scale` on validation
  funded → weighted CDR MAE 0.0207 → 0.0150.
- **WS3 — A 90% PD band is now split-conformal** (`calibration.fit_pd_band`),
  conformity measured on the RAW point (a fitted recalibrator under-covers OOF:
  raw→0.875 vs isotonic→0.53). Binned coverage 0.70 → **0.900** at width 0.133.
- **WS4 — C is g-computation-in-spirit**; the `closed-loop-default-detection`
  harness certifies it beats naive conditioning on the strong-propagation slice
  across a 5-seed counterfactual sweep (seeds 7/13/42/101/2026, 900 queries each):
  at severity 0.4, MAE **0.0797 ± 0.0135** vs naive **0.0988 ± 0.0154** — gap
  +0.0191 ± 0.0046, positive on **5/5 seeds, no sign flips** (~19% relative).
  The advantage has a regime boundary: at full severity the gap is +0.0021 ±
  0.0022 with a sign flip on one seed — statistically zero, so we claim nothing
  there. One disclosed trade-off: g-computation's bias is consistently *more*
  negative than naive's (5/5 seeds; e.g. −0.0252 vs −0.0201) — it buys MAE at the
  cost of slightly worse systematic underestimation.
- **WS5 — Deliverable D** is `submission/submission_D_writeup.md`, auto-filled from
  the artifacts (scorecard, `pnl_backtest.png`, `compute_curve.png`).
- **Compute scaling**: `scripts/run_compute_curve.py` → `reports/compute_curve.csv`;
  weighted_proxy rises monotonically low→med→high (0.4734→0.4912→0.4983 at fixed
  write_frac), diminishing returns med→high. All deterministic per (seed, budget).

> The module-map line in `CLAUDE.md` ("policy = NPV at **upper** PD bound") now
> describes only the `rule="flat"` baseline; the shipped rule is `rule="timing"`
> at the point E[NPV] bound.

> **Merge note (WS1–WS5 → master).** This addendum's WS1–WS5 now sit on master's
> selective-label-safe outcome model (`prior_underwriter_score` excluded, IPW
> removed). The headline `0.3499 → 0.5583` is measured against the *original*
> Step-0 baseline; the incremental delta over the current `master` submission is
> re-measured in `README.md` (master already carried an isotonic OOT recalibration,
> so part of the WS2 trajectory gain was independently present there).

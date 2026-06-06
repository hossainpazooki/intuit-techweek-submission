# Methodology & Writeup Plan — SMB Underwriting Challenge

This document is our methodology of record. It is written to be **judge-ready and
honest**: it separates what this repository *actually does today* from what we
*propose* as the modeling approach, and it states the identification limits of the
problem rather than papering over them. Every empirical number below was recomputed
from the released CSVs (see `scripts/eda.py`), and the technical claims were
adversarially cross-checked before being written here.

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

It does **not** currently implement survival modeling, IPW, doubly-robust
off-policy evaluation (DR-OPE), Double ML (DML), causal forests, a Markov
missed-draw simulation, conformal calibration, or DeepHit. Those appear here as
**proposed** method tiers, not as completed work. (See §10 for the exact
implemented-vs-proposed inventory; the modeling modules are documented stubs.)

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
are none at the same score).

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
outcome model (it is informative only for the funding/propensity model).

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
point effect as if it were a solved causal quantity.

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

- **Calibration is primary.** Decisions flip in a narrow band near PD\* ≈ 8%, so we
  isotonic-recalibrate on a held-out **out-of-time** fold and report reliability
  curves + ECE. **[BASELINE]**
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

**Implemented today — runs [VALIDATED]:**

- `scripts/eda.py` — exploration/validation pass (split shapes; 13,306 decision set;
  selective-labels gap; deterministic funding-threshold check; default & payoff
  timing; recovery distribution; bank-feed gating; intervention-query structure;
  overlap diagnostics; cohort sizes).
- `src/smb/config.py` — loan-economics and cohort constants (constants only).
- Guardrail helpers: `survival.enforce_monotone`, `calibration.clip_and_order`,
  `calibration.coverage` (pure-numpy).
- `scripts/validate.py`, `scripts/run_all.py` — wrappers around the official
  `validate_submission.py` (note: `run_all` cannot complete end-to-end yet because
  the modeling pipeline is stubbed).

**Proposed / not yet implemented (documented stubs):**

- Data loading & label construction, feature building, the PD model, the NPV policy,
  IPW / selection correction, the competing-risks hazard, counterfactual estimation,
  conformal intervals, DR-OPE, and the A/B/C submission builders. The modeling
  modules currently raise `NotImplementedError`; advanced-method names appear only in
  docstrings/TODOs. No modeling libraries (sklearn/lightgbm/lifelines/econml/…) are
  imported yet.

**The submission must keep this implemented-vs-proposed boundary visible.** Use "we
validate," "we propose," "we use as baseline," and "future extension" — never claim
the research stack is built.

---

## 11. Open TODOs

1. Decide and **state** the interest convention (full-principal vs amortizing).
2. Pick a central recovery figure (≈ 0.09R) and disclose dispersion.
3. Implement the **[BASELINE]** tier end-to-end so `run_all.py` produces a
   validator-passing A/B/C (per the approved implementation plan).
4. Distill §1–§8 into the 4-page Deliverable D writeup
   (`submission_D_writeup_template.md` → PDF), leading §3 (causal) with the
   identification story.

<!--
  DELIVERABLE D - TECHNICAL WRITEUP (final draft, export to submission_D_writeup.pdf)

  Conforms to the official template: the five required section headers, in order;
  body <= 4 pages at >= 11pt / >= 0.75in margins; no executive summary. Every
  empirical figure is recomputed from the released CSVs (scripts/eda.py); claims
  are matched to the code (METHODOLOGY.md, docs/VERIFICATION.md). Honesty
  convention: we mark what is implemented and validator-passing vs what we propose
  as a future extension and did not build.

  Team name placeholder: "Hossain Pazooki" - edit before upload if different.
-->

# Deliverable D - Technical Writeup

**Team:** Hossain Pazooki

*What ships vs. what we propose (read first).* Our submission is a **sklearn-only
baseline that passes the official validator**: a weekly **competing-risks
discrete-time hazard ensemble** (`HistGradientBoostingClassifier`, 5 seeds) → a
calibrated lifetime PD → an **upper-bound NPV approval rule** (A), a cohort
cumulative-incidence trajectory (B), and `do()` counterfactuals with structural
propagation (C); intervals are **ensemble percentile bands**, recalibrated
**out-of-time** with isotonic regression. The doubly-robust off-policy, IPW
reweighting, split-conformal, and DeepHit machinery discussed below is **proposed
and was not built** — and we explain why one of them (IPW) is not merely unbuilt
but *unidentifiable* here. The strongest contribution is not model weight; it is a
disciplined account of *what is identifiable, what is partially identifiable, and
what is pure extrapolation* in this data, and a baseline that acts accordingly.

## 1. Problem framing & assumptions violated

The lender maximizes realized portfolio value. With no binding capital constraint
the objective is separable across applicants, so the optimal policy is **pointwise
on the sign of expected NPV**, `d*(x) = 1{E[NPV | x, do(fund)] > 0}`. The payoff is
**sharply asymmetric**: a repaid loan returns single-digit percent of principal
while an early default with low recovery destroys most of it. The task is therefore
to estimate a **calibrated expected NPV**, not to maximize classification accuracy —
decisions flip in a narrow band, so a few points of miscalibration destroy value.

Four data realities — all recomputed from the CSVs — break standard assumptions:

**(a) Selective labels with a deterministic funding rule.** Repayment outcomes
exist only for funded, matured loans (**60.6%** of train, 51,722/85,340). The legacy
funding decision is a **perfect deterministic threshold** on an observed score:
`prior_decision == 1 iff prior_underwriter_score >= ~0.273`, with **zero mismatches
across all 85,340 rows** and a clean gap (max declined 0.27297 < min approved
0.27301). So the funding propensity `e(x) = P(fund | x)` is **degenerate (0/1) and
positivity fails globally** — there is no overlap region. This is the spine of the
submission and is developed in §3.

**(b) Out-of-time deployment.** Training spans 2024-01 to 2025-06; the decision
population (validation + test, **13,306** applicants) is a single later 13-week
window (2025-06-30 to 2025-09-28). We evaluate **out-of-time** on the validation
split, mirroring deployment — not a random split. Validation labels also exist only
for its **funded** subset (2,551/4,489), so even calibration data is policy-selected.

**(c) Path-dependent timing with a point mass.** Default triggers on missed-draw
rules or a nonzero balance at day 90. Verified timing: all paid-in-full loans repay
at **exactly day 60**; defaults occur over days 3–60 **and then jump to a point mass
at exactly day 90** (**22.5%** of defaults), with **zero defaults in (60, 90)**. A
single life-of-loan flag discards timing that both the NPV and Deliverable B need.

**(d) Informative missingness (MNAR by design).** The bank-feed block is null *iff*
`has_linked_bank_feed == False` (exact across all rows); `days_since_*` nulls mean
"no prior event." We preserve missingness as signal (indicator flags + native NaN
handling), never blind-impute.

## 2. Methodology

**One fit, three deliverables.** Rather than regress NPV directly (high variance,
opaque), we model only its stochastic inputs and compose them through the known
cash-flow algebra. A **single weekly competing-risks discrete-time hazard** over the
two absorbing events — **default** vs **payoff** — is fit on funded+matured loans in
person-period layout (`HistGradientBoostingClassifier`, native NaN + sample-weight,
5 seeds bagged). Survival `S(t)=∏_{s≤t}(1−h_d−h_p)` and cumulative incidence
`CIF_d(t)=∑_{s≤t} h_d(s)S(s−1)` give lifetime `PD = CIF_d(H)` for A and the full
trajectory for B off **one coherent fit**. Carrying the payoff hazard prevents
overstating PD (a loan that pays off is removed from default risk). The economics:
amount `R`∈$5–50K, 60-day term, 35% APR, 3% fee, 90-day default window →
`E[NPV|x] = (1−π)(F + interest) + π·E[recovery − loss]`, with `F = 0.03R`.

**Interest convention, disclosed not assumed.** Reading "35% APR" as interest on
full principal for the 60-day term gives `R·0.35·60/365 ≈ 0.0575R`, so a repaid loan
returns `F + interest ≈ 0.0875R`; the amortizing (declining-balance) reading roughly
halves the interest. We **state the full-principal convention explicitly** because
the headline break-even PD swings materially between them.

**Break-even is a band, not a point.** `NPV(π)` is linear and strictly decreasing in
`π`; setting it to zero, `π* = g/(g+1−rec)`. With `g=0.0875`: zero recovery gives
**π\* ≈ 8.05%**, empirical recovery (≈0.091R) gives **≈8.8%** — a band **≈8.0–8.9%**
(the amortizing reading gives ≈5.5–6.1%). The funded-book default rate **17.45%**
sits well above this band, confirming the legacy book funds many negative-NPV loans —
the headroom our policy targets.

**Decision rule — fund on the conservative bound.** Because NPV decreases in PD, the
pessimistic case is the **highest** credible default rate, so we **approve only if
NPV stays positive at the UPPER 90% PD bound**. Uncertain loans push their
upper-bound PD past break-even and **default to denial** — ignorance is denied, not
approved (the inverse rule would be backwards). On the 13,306 applicants this yields
**7.2% approval** (957 loans), approved-set mean PD **0.041** with **every approval's
upper-bound PD < 0.084 < break-even**, vs declined-set mean PD **0.28**.

**Feature hygiene.** The outcome model excludes `prior_underwriter_score` (§3),
`prior_approved_amount` (funded-only leakage), and the constant `prior_decision`;
features are kept **numeric with NaN preserved** (no category dtype) to avoid
cross-split alignment bugs. Recovery is an **empirical mean ≈0.091R** with disclosed
dispersion (median 0.072, ~23% recover exactly 0), not yet a censoring-aware
regressor. *Proposed, not built:* a broader LightGBM/XGBoost/logistic model family
(the stack is deliberately sklearn-only for runtime portability) and a Markov
missed-draw simulation of the exact default automaton.

## 3. Causal reasoning & counterfactual methodology

Deliverable C asks for an **interventional** quantity,
`P(default | do(f=v), X_{−f}=x_{−f})`, which differs from the **observational**
`P(default | f=v, …)` whenever `f` is confounded. Re-scoring a fitted predictor with
one feature swapped — and, equivalently, reading a SHAP attribution — answers the
*observational* question and is **not** a `do`-operator estimate. We state this
plainly rather than dressing a model perturbation as a causal effect.

**The identification problem is the core finding.** Repayment is observed only on the
historically funded set `S`, selected by the legacy policy. We verified that policy
is a **deterministic threshold** on `prior_underwriter_score` (0 mismatches/85,340;
no overlap), so the funding propensity is **degenerate and positivity fails
globally**. The consequences are concrete and we follow each to its honest end:

- **Covariate-shift / IPW reweighting toward the applicant marginal is not
  identified** — there are no funded look-alikes at a declined applicant's score to
  reweight. We therefore **removed the IPW / reject-inference path entirely** and fit
  the hazard **unweighted on funded labels only**; `propensity.py` survives purely as
  a diagnostic that *proves* the rule and reports positivity. Demoting IPW from "a
  thing we do" to "the failure mode that forces abstention" is deliberate honesty.
- **`prior_underwriter_score` is excluded from the outcome model.** Funded rows only
  ever carry `score ≥ 0.273`, yet **43.5%** of the decision population sits *below*
  that minimum — the score is out of training support for nearly half the applicants
  we must decide on. Using it would extrapolate a legacy-policy artifact across the
  funding boundary and bake selection into the default model.
- **Outside funded support we do not extrapolate silently.** Any cross-threshold
  prediction for a declined-type applicant is an explicit modeling extrapolation
  (the default function assumed smooth across the boundary); we widen intervals, use
  **Manski-style partial-identification** language, and abstain where appropriate.

**What the baseline does honestly for C.** Where a dependency is *deterministic* we
**propagate structurally**: an intervention on `has_linked_bank_feed` propagates to
the entire bank-feed block (verified strict gating), not a lone boolean; interventions
on `requested_amount` / revenue propagate to the engineered
`requested_amount_to_observed_revenue` ratio. These are mechanical, defensible edits.
**Everything else is reported as the model's observational response under feature
substitution, with the observational-vs-interventional gap surfaced, not hidden.**
The query set is 900 queries / 300 applicants / 30 features; we distinguish
**hard-structural identity** features (sector, region, vintage, employee bucket —
not coherently intervenable) from **merely-not-actionable history** (account age,
prior loan counts). For confounded drivers like bank-feed linkage we make the
confounding case via **covariate association** (linkers vs non-linkers differ in
credit utilization ≈0.40 vs 0.47, requested amount, and underwriter score), **not**
via raw default rates — which are nearly identical (0.207 vs 0.206) and would
*understate* confounding. C intervals are correspondingly **wider than A**
(mean width 0.091 vs 0.070), reflecting that interventions are less identified.

**Regulatory defense of stated drivers.** For each driver we would argue **sign,
magnitude, and mechanism**, and explicitly flag proxies as associational unless
backed by the structural-propagation logic above. The honest regulator-facing claim
is: "this feature moves the model's *predicted* risk by X; we can defend it as a
*causal* lever only where the dependency is mechanical (bank-feed gating, ratio
recomputation); elsewhere it is an association, and for applicants below the legacy
funding threshold it is an extrapolation we bound rather than assert." *Proposed, not
built:* a backdoor-adjusted SCM / Double-ML / causal-forest estimator — under global
positivity failure several interventions are only partially identified, so we
prioritized stating the gap over shipping a point effect that looks solved.

## 4. Calibration & uncertainty quantification

**Calibration is the scoreboard, and is applied out-of-time.** Because decisions
flip in a narrow band near `π* ≈ 8.8%`, ranking is secondary to calibration. We fit
an **isotonic** recalibrator on the **validation funded subset** (n=2,551 — its
realized lifetime default, scored by the train-fit ensemble, a genuinely
out-of-time check) and **apply it to the submitted A/B/C PDs**. The map is monotone,
so it preserves `lower ≤ point ≤ upper` and B's per-cohort age-monotonicity, and B is
calibrated with the **same** map as A so the cross-check `cdr@age13 ≈ approved-set PD`
holds (0.047 vs 0.041). We gate the application on a **K-fold cross-fitted** ECE
(held-out, never isotonic's degenerate in-sample ≈0): out-of-time ECE improves
**0.0228 → 0.0199** and mean predicted PD rises **0.187 → 0.206** to match realized
**0.206**. The decision effect is large precisely because the model under-predicts
*near break-even*, which tightens approvals from **~21% (uncalibrated) to ~7%** — the
correct conservative consequence of honest calibration. **Caveat, stated:** the
calibration slice is *funded*, so applying the map across the whole population rests
on a smoothness/extrapolation assumption — concentrated where it is most trustworthy
(the in-support, near-break-even region).

**Intervals.** 90% bands are the **5th/95th percentiles of the 5-seed hazard bag**
for A, B, and C. We **fund on the conservative bound** (upper PD = lower NPV), so
ignorance defaults to denial and bands widen near the support boundary. We chose
ensemble dispersion over a single point because the binding uncertainty here is
*model/extrapolation* uncertainty across an unobserved region, which seed dispersion
exposes; the trade-off is that these bands are **honest-but-not-coverage-guaranteed**.
*Proposed, not built:* **split-conformal** intervals for finite-sample coverage
(`calibration.conformal_intervals` exists but is unused) — the principled upgrade,
though under the positivity failure even conformal coverage holds only in-support.

## 5. Limitations & what we'd do differently

- **Identification, not estimation, is the binding constraint.** The deterministic
  funding rule means no reweighting recovers declined-applicant outcomes; we bound
  and abstain rather than ship a confident-looking but unidentified IPW/DR number.
  This is the submission's deliberate stance, not an omission.
- **The calibration and counterfactual maps lean on a smoothness assumption** across
  the funding boundary (positivity fails). With another day we would quantify its
  fragility — sensitivity of approvals to the assumed cross-boundary slope, and
  Manski bounds on the approved-set value.
- **Recovery is observed only on a selected sliver** of defaulters and modeled as a
  shrunken mean; a censoring-aware recovery regressor is the natural next step.
- **Intervals are ensemble percentile bands, not conformal** — coverage is plausible,
  not guaranteed; split-conformal (in-support) is the first upgrade.
- **Off-policy value is only partially identified** here; a doubly-robust `V̂(d)` with
  bootstrap CIs and a Rosenbaum sensitivity check is the evaluation we would *report
  as bounds*, not a single number, and did not build.
- **Stack is deliberately sklearn-only** (HistGradientBoosting + IsotonicRegression,
  pandas/numpy); the hazard spine is model-agnostic, so swapping in a richer learner
  family (LightGBM/XGBoost/logistic stack) is a clean, proposed upgrade path.

What we are most confident in: the data diagnosis (every empirical figure here is
recomputed from the raw CSVs and survived an adversarial verification pass), the
decision-rule direction (conservative upper-bound), and that A/B/C pass the official
validator (`RESULT: PASS`, 0 errors). What we are least confident in: any prediction
for applicants below the legacy funding threshold — which is exactly why we bound it.

<!-- References (do not count toward the 4-page body limit). -->
**References.** Lakkaraju et al. (2017), *The Selective Labels Problem*, KDD.
Kozodoi et al. (2025), *Fighting Sampling Bias*, EJOR 324(2):616–628. Lessmann et al.
(2015), EJOR 247(1):124–136. Verbraken, Verbeke & Baesens (2013), *Profit-maximizing
metric*, IEEE TKDE 25(5):961–973. Bellotti & Crook (2013), *Forecasting and stress
testing credit card default*, IJF 29(4):563–574. Angelopoulos & Bates (2023),
*Conformal Prediction: A Gentle Introduction*, FnT ML 16(4):494–591. Athey & Wager
(2021), *Policy Learning With Observational Data*, Econometrica 89(1):133–161.
Lundberg & Lee (2017), *A Unified Approach to Interpreting Model Predictions*, NeurIPS
(with the SHAP non-causal caveat). Pearl (2009), *Causality* (2nd ed.), Cambridge UP.

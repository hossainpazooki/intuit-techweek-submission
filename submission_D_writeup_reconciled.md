<!--
  RECONCILED one-page solution write-up.

  Source: Hossain_Pazooki_Lending_Challenge_Solution.pdf (the uploaded 1-pager).
  This version keeps that document's 7-section structure but (a) tags every
  section with its honest implementation tier and (b) corrects claims that
  overstate what the repo actually does, so the narrative matches the code.

  Tier legend (same as METHODOLOGY.md):
    [VALIDATED] implemented + runs today (guardrail/exploration)
    [BASELINE]  implemented end-to-end; what we submit (sklearn/pandas stack)
    [STRETCH]   proposed; a reach within the window
    [FUTURE]    research-grade; NOT implemented, not claimed as built

  Grounding: src/smb/{pipeline,calibration,policy,survival,propensity}.py,
  METHODOLOGY.md (methodology of record), docs/VERIFICATION.md.
-->

# Counterfactual Policy Learning for Small-Business Lending — reconciled

Hossain Pazooki | Explainable ML Prediction Challenge | github.com/hossainpazooki

> **Implemented-vs-proposed boundary (read first).** What ships today is a
> **sklearn-only baseline**: a weekly **competing-risks discrete-time hazard
> ensemble** (`HistGradientBoostingClassifier`, 5 seeds) → lifetime PD →
> **upper-bound NPV approval rule** (A), cohort CIF trajectory (B), and `do()`
> counterfactuals with structural propagation (C); intervals are **ensemble
> percentile bands**; `run_all.py` → official validator **`RESULT: PASS`**.
> The reweighting, conformal, and doubly-robust machinery below is **proposed**,
> not built. Tags mark exactly which is which.

## 1. Objective ⇒ decision rule. **[BASELINE]**
We choose an approval policy `d : X → {0,1}` to maximize realized portfolio value
`max_d E[Σ_i d(x_i)·NPV_i]`. With no capital constraint the objective is separable,
so the optimum is pointwise: `d*(x) = 1{ E[NPV | x, do(fund)] > 0 }`. The problem
reduces to estimating a **calibrated** conditional expected NPV on the applicant
population — not to maximizing classification accuracy. *(Implemented as the
upper-bound NPV rule in §5; `policy.expected_npv` / `policy.decide`.)*

## 2. Structured NPV model. **[BASELINE]** (timing model: implemented)
Rather than regress NPV directly, we model only its stochastic inputs and compose
them through the known cash-flow algebra:
`E[NPV|x] = (1−π(x))(F + R·r·T/365)  +  π(x)·E[F + D(t*−1) + rec − R | x, default]`,
with `F = 0.03R` and a repaid-loan return ≈ **0.0875R** (full-principal convention,
stated explicitly). Components: **(i)** PD `π(x)` and **(ii)** default timing `t*`
come from a **single weekly competing-risks discrete-time hazard** (default vs
payoff) — *not two separate models* — so A and B read off one fit
(`survival.py`); **(iii)** recovery `rec` is an **empirical mean ≈ 0.091R** with
disclosed dispersion (`recovery.estimate_recovery_rate`), not yet a
censoring-aware regressor. Composing avoids the variance of a single NPV
regression and stays interpretable. *(Correction vs the PDF: the stack is
`HistGradientBoostingClassifier`, **not LightGBM**; recovery is an empirical
mean, not a regressor.)*

## 3. Selective labels (the causal core). **[VALIDATED] diagnosis · reweighting is [FUTURE]**
Repayment is observed only for historically funded loans `S`, selected by the
legacy policy. We verified that policy is a **deterministic threshold** on
`prior_underwriter_score` (`prior_decision == 1 iff score ≳ 0.273`, **zero**
mismatches across 85,340 rows; max declined 0.27297 < min approved 0.27301). So
the funding propensity `e(x)` is **degenerate (0/1) and positivity fails
globally** — there is *no* overlap region. **Consequence (corrected vs PDF):**
covariate-shift reweighting toward the applicant marginal is **not identified**;
the IPW / reject-inference reweighting path is therefore **removed** and the hazard
is fit **unweighted** on funded labeled data only (`propensity.py` is now a pure
diagnostic that *proves* the rule). The same logic excludes
`prior_underwriter_score` from the **outcome** model — funded rows only ever have
`score ≥ 0.273` while **~44%** of the decision population sits below that minimum, so
the score is out of training support there; we keep it only for the funding-rule
diagnostic. Outside funded support we **do not extrapolate silently**: we use wider
bands, **Manski-style partial-identification** language, and **abstention**. *(The
PDF frames reweighting as something we perform; here it is correctly demoted to the
failure mode that motivates abstention.)*

## 4. Calibration is the scoreboard. **[BASELINE] — applied out-of-time**
The payoff is sharply asymmetric (repaid ≈ +0.0875R; early default with low
recovery ≈ −0.97R), so decisions flip in a **narrow band** around the break-even
PD and a few points of miscalibration destroy value. **Break-even (corrected):**
under the full-principal convention the **true worst case (zero recovery) is
π\* ≈ 8.05%**, with an operative **band ≈ 8.0–8.9%** (empirical recovery ≈0.09R);
the amortizing reading gives ≈5.5–6.1%. We report a **band, not a single
"8.3%"** (the funded-book default rate ≈17.45% sits well above it). We now **apply
isotonic recalibration to the submitted A/B/C PDs**, fit **out-of-time** on the
validation funded subset (train-fit ensemble scored on a later cohort; mean
predicted **0.187 → 0.206** vs realized **0.206**) and gated on a **cross-fitted**
ECE improvement (**0.0228 → 0.0199** — never isotonic's degenerate in-sample ≈0).
Because the model under-predicts *near break-even*, this is small in ECE but large
in decisions: approvals tighten from ≈21% (uncalibrated) to **≈7%** — the honest
consequence of calibrating where decisions flip. **Caveat:** the calibration set is
the *funded* slice, so applying the map across the whole population is a stated
smoothness/extrapolation assumption, concentrated where it is most trustworthy.
*(Correction vs PDF: isotonic calibration is now genuinely applied — out-of-time,
not in-sample.)*

## 5. Uncertainty & interventions never observed. **[BASELINE] (ensemble bands) · conformal is [FUTURE]**
Predictions outside funded support are extrapolations. We attach uncertainty via
**ensemble dispersion**: the 5/95 percentiles of the 5-seed hazard bag
(`calibration.ensemble_intervals`), and we **fund on the conservative bound** —
approve only if NPV stays positive at the **upper PD bound** (= lower NPV
confidence bound; `policy.decide(upper, …)`). So **ignorance defaults to denial**
and bands widen near the support boundary. *(Correction vs PDF: shipped intervals
are **ensemble percentile bands, not split-conformal**; `calibration.conformal_intervals`
exists but is unused. "Deep-ensemble" = a 5-seed GBM bag, not neural ensembles.)*

## 6. Defending the policy (off-policy evaluation). **[FUTURE] — NOT implemented**
The doubly-robust off-policy estimator `V̂(d)` with bootstrap CIs and a Rosenbaum
sensitivity check **describes the intended evaluation, not a built component**.
Because the legacy rule is a deterministic threshold (§3), DR/IPW are valid only
on common support, which here is **empty across the threshold** — so the policy
value is **partially identified**; we would report **bounds / abstention**, not a
single DR number. *(Correction vs PDF: this section is aspirational and is tagged
as such; nothing in the repo computes `V̂(d)`.)*

## 7. Validation, leakage, stack. **[BASELINE] split/leakage · [FUTURE] CVaR**
Strictly **out-of-time** evaluation on the validation split with point-in-time
joins — no post-origination signal leaks into features. A CVaR tail-loss cap is a
**proposed** risk-adjusted extension `[FUTURE]`. **Stack (corrected):**
`HistGradientBoostingClassifier` (PD) + a **weekly competing-risks discrete-time
hazard** (timing) + ensemble percentile intervals, in a reproducible
**pandas/numpy** pipeline. *(Correction vs PDF: **sklearn-only, not LightGBM**;
**pandas, not polars**; isotonic/conformal/DR are utilities-or-proposals as tagged
above.)* Engineering effort goes into correctness and calibration, where the score
is won.

---

### Implementation status (mirrors METHODOLOGY.md §10)
**Built & validator-passing:** data/features/survival/recovery/policy/dag/causal/
calibration/pipeline; competing-risks hazard ensemble (fit **unweighted**, no IPW;
`prior_underwriter_score` excluded) → **out-of-time isotonic calibration** → A/B/C;
`run_all.py` → `RESULT: PASS`; sklearn + pandas only.
**Proposed / NOT built:** the broader **LightGBM/XGBoost/logistic** model family
(sklearn-only by design), DR-OPE, Double ML, causal forests, split-conformal
intervals, Markov missed-draw simulation, DeepHit. IPW reweighting is **removed**
(not merely unused) — positivity fails, so it is not identified.

> **Format note (not a content change):** the official Deliverable D template
> (`submission_D_writeup_template.md`) enforces **5 fixed section headers** and a
> 4-page limit. This 7-section narrative does not match that structure — if the
> PDF is the graded artifact, it should be remapped onto the required headers.

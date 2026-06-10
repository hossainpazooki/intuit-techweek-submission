# Adversarial Verification of the Methodology

Before any of the revised methodology was written into the repo, every technical
and empirical claim was put through an **adversarial verification pass**: a
multi-agent workflow whose design principle is that *a claim is only "correct" if
it survives an independent attempt to break it*. The corrected claims — not the
prior draft's errors — are what got written into [`METHODOLOGY.md`](../METHODOLOGY.md)
and built into the pipeline.

## Method

A background workflow of **19 agents** structured as a three-phase
fan-out → refute → synthesize pipeline, followed by an independent human-driven
spot-check of the single most load-bearing claim.

### Phase 1 — Check (9 parallel checkers, one per dimension)

Each checker was scoped to a single dimension and forced to return **structured**
findings (schema: `claim / verdict ∈ {correct, incorrect, partially_correct,
uncertain} / evidence / correction`) so verdicts could not be vague. The prompt
forbade accepting any claim without a **derivation**, a **quote from the brief**,
or a **figure recomputed from the raw CSVs with pandas**. Dimensions:

1. Decision-rule direction (upper vs lower PD bound)
2. NPV economics arithmetic
3. Timing horizons (60-day interest vs 90-day default observation)
4. Deliverable B semantics (cohort CDR vs defaulter-normalized vs lifetime PD)
5. Deliverable C semantics (`do()` vs observational; structural propagation)
6. Selective labels & overlap (IPW / DR identifiability)
7. All empirical data facts (recomputed from `dataset/*.csv`)
8. Repo honesty (stub-vs-implemented audit)
9. Citations (real authors / years / venues, via web search)

### Phase 2 — Verify (adversarial skeptics)

Every finding **not** already marked `correct` was handed to a fresh skeptic agent
with an explicit mandate: *try to refute this; only uphold it if it genuinely
survives.* Each skeptic returned whether the original verdict held and the
statement that is *actually* correct. This is what caught the interest-model
nuance and the mislabeled "worst-case" break-even.

### Phase 3 — Synthesize

A lead agent consolidated everything into a correctness report: confirmed claims,
required corrections, a pass/fail data-facts table, implemented-vs-proposed lists,
and final citation strings — surfacing disagreements rather than papering over them.

### Phase 4 — Independent spot-check

The workflow's conclusions were not taken on faith. The single load-bearing claim
(the deterministic legacy funding rule) was re-run by hand and found to be **even
stronger** than reported.

## Findings

### Confirmed — 8/8 empirical facts passed (recomputed from the data)

- Funded fraction **60.6%** (51,722 / 85,340).
- Default rate among matured loans **17.45%**.
- Recovery (`final_recovered_amount / requested_amount`) mean **0.091**, median
  **0.072**, **23%** recover exactly 0.
- Bank-feed block null **iff** `has_linked_bank_feed == False` (perfect gating).
- `prior_decision` constant (`== 1`) within the labeled set.
- Intervention queries: **900 / 300 applicants / 30 features**, ~19% structural.
- Validation + test cohort partition totals **13,306**, fully assigned to weeks 1–13.
- Paid-in-full loans repay at **exactly day 60**.

### Corrections forced before writing

| # | Wrong (prior draft) | Corrected |
|---|---|---|
| 1 (critical) | Use the **lower** PD bound; "lower bound makes ignorance default to denial" | NPV is strictly decreasing in PD ⇒ approve only if NPV > 0 at the **upper** PD bound. The lower bound is the *optimistic* case and would make ignorance default to *approval*. |
| 2 | "8.3% worst-case break-even" | Worst case (zero recovery) = **8.05%**; band ≈ **8.0–8.9%** under the full-principal convention. |
| 3 | Interest treated as settled | **Disclose the convention**: full-principal (~0.0875R, break-even ~8%) vs amortizing (~0.059R, break-even ~5.5–6.1%). |
| 4 | B's CDR conflated with defaulter-timing / lifetime PD | CDR is over **all approved** cohort loans; at age 13 ≈ approved-cohort lifetime rate (selection-dependent), **not 1.0, not the 20.6% book rate**; `CDR(a) = cohort_rate × defaulter_normalized(a)`. |
| 5 | "validation has outcomes filled in" | Only the **funded subset** (2,551 / 4,489) is labeled; calibration on validation is itself conditioned on the legacy policy. |
| 6 | `prior_decision == 'approved'` (string) | Integer encoding `prior_decision == 1`. |
| 7 | Citations | Three mis-attributions fixed: Verbeke dropped from the 2014 credit paper; the IFRS-9 discrete-survival framing detached from Bellotti & Crook (anachronistic); Generalized Random Forests keeps Tibshirani. |

### The standout finding

The legacy funding policy is a **perfect deterministic threshold** on
`prior_underwriter_score` (≈ 0.273): **zero** mismatches across all 85,340 rows,
with a clean gap (max declined score `0.27297` < min approved score `0.27301`) and
**no overlap whatsoever**.

Consequence: the funding propensity `e(x) = P(funded | x)` is **degenerate (0/1)**,
so **positivity fails globally**. IPW and doubly-robust estimators are therefore
**not identified** for the declined region — reweighting cannot recover the outcomes
of applicants the legacy policy never funded (there are no funded look-alikes at the
same score). Partial identification / abstention is not a stylistic choice; it is
the only defensible stance.

This finding flows directly into the implementation: the hazard model is fit
**unweighted** (no IPW), and `src/smb/propensity.py` exists for positivity
diagnostics only — exactly so the submission does not overclaim an identification it
cannot support.

## Provenance

- Correctness workflow run ID: `wf_397d1ec5-5fd` (19 agents).
- A separate 13-agent workflow implemented the baseline pipeline; its output was
  then independently re-run and validated (official validator → `RESULT: PASS`,
  0 errors) before commit. See [`METHODOLOGY.md`](../METHODOLOGY.md) §10 for the
  implemented-vs-proposed boundary.

## Round 2 — the synthetic-harness results (2026-06-10)

The sibling repo (`closed-loop-default-detection`, harness master `283a040`)
hosts the synthetic world built to study the same selective-labels mechanism
this submission hit on the real data. Its headline numbers were put through the
same discipline — reproduce first, then try to break it — before being quoted
anywhere. **None of this touches the real-data findings above.** The positivity
failure and the IPW-not-identified conclusion are about the hackathon dataset
and stand unchanged; the harness results are about a synthetic world where the
selection severity is a knob, not a fact.

### Claim 1 — the g-computation advantage (survived in part, refuted in part)

**What was claimed.** A single-seed result that g-computation beats naive
conditioning on counterfactual MAE, from the harness's own
`scripts/run_scorecard.py` c_proxy config (n_applicants=5500,
n_query_applicants=200, seed 42, severity 0.4, `--compute high`).

**How it was attacked.** Reproduce, then de-seed. The scorecard figure
reproduces bit-for-bit on current harness master **per environment, and the
environment matters at the third decimal**: this repo's Python (sklearn 1.8.0,
which built the committed `reports/scorecard.json`) yields gcomp 0.0854 vs
naive 0.1085; the harness venv (sklearn 1.9.0) yields 0.0869 vs 0.1087 for the
identical call — HistGradientBoosting differs across sklearn releases. Neither
is wrong; both are single-seed and environment-pinned, which is exactly why no
load-bearing claim rests on this figure. The skeptic move was a 5-seed sweep
(seeds 7/13/42/101/2026, 900 Deliverable-C-style queries each) at both
severities, looking for sign flips.

**What survived.**

- **Severity 0.4, strong-propagation slice: the advantage is real.** G-comp MAE
  **0.0797 ± 0.0135** vs naive **0.0988 ± 0.0154**; gap **+0.0191 ± 0.0046,
  positive on 5/5 seeds, no sign flips** — a ~19% relative reduction. The
  overall (all-query) gap is **+0.0031 ± 0.0025**, much thinner; the win lives
  where interventions actually propagate.
- **Seed 42 — the previously published seed — turned out to be the most
  pessimistic of the five.** The single-seed number *understated* the
  strong-propagation advantage rather than cherry-picking it.
- **The full-severity advantage did NOT survive.** At severity 1.0 the
  strong-propagation gap is **+0.0021 ± 0.0022 with a sign flip on seed 13** —
  statistically zero. Any claim of even a small full-severity win died here;
  the docs say "no reliable advantage" and nothing softer.
- **A disclosed trade-off, not seed noise:** g-comp's bias is *more* negative
  than naive's on **5/5 seeds** at severity 0.4 (seed 42: −0.0252 vs −0.0201).
  MAE improves while systematic underestimation worsens slightly.

### Claim 2 — the unified-world frontier (survived)

**What was claimed.** The selective-labels operating frontier (IPW holds
declined-cohort calibration through severity 0.4, fails at 0.6) had been
measured in a *different* synthetic world (the flat generator) than the
counterfactual results (the SCM) — two worlds, one narrative, a refutable gap.
The fix under test: `SelectiveLabelsLoop` now runs on the SCM itself
(`generator="scm"`), so both failure modes are measured in the same world.

**How it was verified.** The change had to prove it altered nothing it didn't
own: the default RNG path is **sha256-verified identical** (same cohorts,
checked across processes), the flat-generator baseline is **byte-identical**
(frozen-baseline test with exact float equality), **50/50 tests** pass, and the
SCM fidelity gate is **51/51 checks green**.

**What survived.** SCM frontier, seed 42: IPW declined-ECE **0.0251 / 0.0370 /
0.0874** at severity 0 / 0.2 / 0.4 (pass), **0.2498** at 0.6 (fail) → the
operating frontier lands at **severity 0.4, the same frontier as the flat
world**, now measured in the same synthetic world as the counterfactual
results. The unified claim this licenses: inside the frontier (severity ≤ 0.4)
IPW holds declined-cohort calibration *and* g-computation reliably improves
counterfactual MAE; beyond it, selection on an *unobserved* confounder defeats
both — one structural mechanism, two measured failure modes.

### The standout catch — a shared exogenous draw

Recon *before* implementation caught that the SCM's selection blend reused the
exogenous draw behind the **observed** `prior_underwriter_score` column
(corr ≈ 0.92 with the selection score at severity 0; an in-sample propensity
model reached AUC ≈ 1.0 on what was supposed to be selection-at-random). On the
flat generator, severity 0 means selection-at-random that *no* propensity model
can explain — so pointing the loop at the SCM naively would have **silently
inverted the severity semantics** and made the two frontiers incomparable, with
every downstream number still computing happily. Fixed with a gated
`independent_selection_noise` flag (default off): a dedicated frozen
selection-noise node drawn after all existing draws, so the default RNG stream
is sha256-identical and the fidelity gate stays 51/51 green.

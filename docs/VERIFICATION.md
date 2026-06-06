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

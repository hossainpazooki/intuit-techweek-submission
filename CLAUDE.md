# CLAUDE.md

Guidance for Claude Code working in this repo (Intuit TechWeek SMB Underwriting
Challenge submission). Read this first; it encodes how I want you to work here.

## Where this runs (read before anything else)

I use Claude Code **from the web** (Linux cloud), and also locally on **Windows**.
The pipeline itself is **platform-agnostic** — it runs unchanged on either. Only a
couple of notes below are Windows-local workarounds; they are explicitly tagged
**[Windows-local only]** and you can ignore them on the web/Linux. When in doubt,
assume Linux and plain `python`. Because I'm usually on the web, **there is no local
fallback — always commit and push your work to `origin`** (see Git below).

## What this repo is

Our solution to the SMB Underwriting Challenge. Authoritative docs:
- [`README.md`](README.md) — unified entry point: challenge framing, method, results,
  learnings, run + submit instructions (the challenge brief is folded in here).
- [`METHODOLOGY.md`](METHODOLOGY.md) — methodology of record (tiered, honest).
- [`docs/VERIFICATION.md`](docs/VERIFICATION.md) — adversarial verification record.
- [`LEARNINGS.md`](LEARNINGS.md) — non-obvious patterns, gotchas, operational
  knowledge (read this before touching the pipeline — it lists the validator traps,
  data realities, and design tricks that will bite you otherwise).
- `dataset/README.md` — upstream Intuit dataset guide.

The modeling spine is a **weekly competing-risks discrete-time hazard** (default vs
payoff); Deliverables A (PD + NPV policy), B (cohort trajectory), C (counterfactuals)
all read off one fit. Baseline tier is implemented and passes the validator.

## Commands

Cross-platform (web/Linux and Windows). On the web, run them exactly as written.

```bash
pip install -r requirements-dev.txt                 # scikit-learn + matplotlib
unzip dataset/dataset-compressed.zip -d dataset/    # if train/validation/test.csv missing

python scripts/run_all.py     # regenerate A/B/C + validate (must print RESULT: PASS)
python scripts/validate.py    # validator only
python scripts/eda.py         # premise validation / exploration
```

**[Windows-local only]** the local Windows console is cp1252, so there I prefix
`PYTHONUTF8=1 python ...` and keep `print()` ASCII-only. On the web/Linux UTF-8 is the
default — **ignore that prefix and the ASCII-print rule**; plain `python` is correct.

## Environment constraints (hard)

- **sklearn-only stack.** Use `HistGradientBoostingClassifier` (native NaN +
  sample_weight) + `IsotonicRegression`; hand-roll anything else. **Do not add**
  lightgbm/xgboost/lifelines/econml/doubleml — they're a `[FUTURE]` tier, not a
  pip-install. (Original reason: the local box is Python 3.14, which has no wheels for
  them. The pipeline imports only sklearn, so it runs on any Python ≥3.11 incl. the
  web runtime — keep it that way so behavior is identical everywhere.)
- Model features are kept **numeric with NaN preserved** (no `category` dtype) to
  avoid cross-split alignment bugs. Keep it that way unless you have a strong reason.

## How I want you to work

### Use the Workflow feature for substantial work
I opt into multi-agent orchestration explicitly (I'll say "use workflows" / "spin up
a team", or you'll see ultracode). When I do:
- **Fan out** independent work as parallel agents, each owning **disjoint files**
  against **one shared interface contract** embedded in the prompt (this is how the
  modules here were built — it prevents interface drift).
- Use `pipeline()` by default; reserve a `parallel()` barrier for when a stage truly
  needs all prior results (e.g. dedup before verify).
- Give each agent a **structured output schema** so results are machine-usable.
- End multi-file builds with a **single integration agent** that runs the pipeline
  and **iterates until the official validator prints `RESULT: PASS`**.
- Do **not** trust a workflow's self-reported success — see verification below.

### Adversarial testing before claiming or writing
This is a standing preference, not a one-off. Before I write a claim into a doc or
trust a result:
- **Recompute every empirical number from the raw CSVs** (don't restate from memory).
- **Try to refute** each technical claim with an independent skeptic; a claim is
  "correct" only if it survives. For verification workflows use the
  fan-out → refute → synthesize shape (see `docs/VERIFICATION.md` for the template).
- **Independently re-run** the load-bearing claims yourself (Bash) rather than relying
  on a subagent's word. The official validator is the final gate — run it, don't
  assume it.
- Default to skepticism over agreement; surface disagreements explicitly.

### Honesty (non-negotiable)
- Keep the **implemented-vs-proposed** boundary visible everywhere. Tag methods
  `[VALIDATED] / [BASELINE] / [STRETCH] / [FUTURE]`. Never present aspirational work as
  built. Use "we validate / we use as baseline / we propose / future extension".
- When the data contradicts a stated assumption (e.g. positivity), say so and adjust —
  don't paper over it. Partial identification / abstention beats a confident wrong
  number.
- If you change repo state in a way that makes a doc stale, update the doc in the same
  change.

### Submission integrity (the validator is law)
- A/B/C are assembled by **LEFT-JOIN onto `expected_ids/*.txt`**; every prob column
  must be **non-null in [0,1]** with `lower <= point <= upper`; B must be the full
  13x13 integer grid and **monotone per cohort**. Never ship a submission you haven't
  run through `validate_submission.py`.

## Git / pushing

- This is a **private** repo; `origin` = `hossainpazooki/intuit-techweek-submission`,
  `upstream` = the official Intuit repo (pull only, never push).
- **I usually run from the web, so there is no local fallback — never leave finished
  work only in the working tree.** At every good checkpoint, commit and **push to
  `origin`**, then confirm sync (`0 ahead / 0 behind`). If a session ends with unpushed
  commits, that work is effectively lost to me. (This is the *opposite* of my other
  repos' "output the command, don't run it" policy — here, you run it.)
- Commit/push when I ask or at a natural checkpoint; branch off `master` for anything
  risky. End commit messages with:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- The unzipped `dataset/*.csv` are gitignored (reproducible from the committed zip);
  don't commit them. `submission/*.csv` ARE committed (versioned deliverables).

## Module map (`src/smb/`)

`config` (constants) · `data` (load/labels/cohort/missingness) · `features` (numeric
matrix + intervenable metadata + propagation) · `survival_data` (person-period) ·
`survival` (hazard ensemble, CIF, B trajectory) · `recovery` · `propensity`
(diagnostics only — IPW NOT used; positivity fails) · `policy` (NPV at **upper** PD
bound) · `calibration` · `dag` + `causal` (do() counterfactuals) · `pipeline`
(assemble + in-process validate).

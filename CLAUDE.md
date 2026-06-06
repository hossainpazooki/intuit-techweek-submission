# CLAUDE.md

Guidance for Claude Code working in this repo (Intuit TechWeek SMB Underwriting
Challenge submission). Read this first; it encodes how I want you to work here.

## What this repo is

Our solution to the SMB Underwriting Challenge. Authoritative docs:
- [`METHODOLOGY.md`](METHODOLOGY.md) — methodology of record (tiered, honest).
- [`README_SOLUTION.md`](README_SOLUTION.md) — orientation + run instructions.
- [`docs/VERIFICATION.md`](docs/VERIFICATION.md) — adversarial verification record.
- [`LEARNINGS.md`](LEARNINGS.md) — non-obvious patterns, gotchas, operational
  knowledge (read this before touching the pipeline — it lists the validator traps,
  data realities, and design tricks that will bite you otherwise).
- `README.md`, `dataset/README.md` — upstream Intuit material (challenge brief).

The modeling spine is a **weekly competing-risks discrete-time hazard** (default vs
payoff); Deliverables A (PD + NPV policy), B (cohort trajectory), C (counterfactuals)
all read off one fit. Baseline tier is implemented and passes the validator.

## Commands (Windows / PowerShell)

```powershell
# regenerate A/B/C and run the official validator (must print RESULT: PASS)
PYTHONUTF8=1 python scripts/run_all.py
PYTHONUTF8=1 python scripts/validate.py     # validator only
PYTHONUTF8=1 python scripts/eda.py          # premise validation / exploration

unzip dataset/dataset-compressed.zip -d dataset/   # if train/validation/test.csv missing
```

## Environment constraints (hard)

- **Python 3.14 on Windows.** Stack is **sklearn-only** by necessity: 3.14 lacks
  wheels for lightgbm/xgboost/lifelines/econml/doubleml. Use
  `HistGradientBoostingClassifier` (native NaN + sample_weight) + `IsotonicRegression`;
  hand-roll anything else. **Do not add those missing deps** — if a method needs them,
  it's a `[FUTURE]` tier, not something to pip-install.
- **Always run python as `PYTHONUTF8=1 python ...`** and keep console prints
  **ASCII-only** (the cp1252 console errors on `pi`, `~`, `<=`, etc.). Use unicode
  freely in `.md` files, never in `print()`.
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
- **I often won't have local access** — when work is at a good checkpoint, commit and
  **push to `origin`**, then confirm sync (`0 ahead / 0 behind`). Don't leave finished
  work uncommitted.
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

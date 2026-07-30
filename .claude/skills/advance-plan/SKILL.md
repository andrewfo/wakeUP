---
name: advance-plan
description: Reconcile docs/PLANNING.md against the actual state of the wakeUp repo, correct any stale statuses, then implement the single next manageable step (with tests) and record it. Use when the user says "advance the plan", "do the next step", "reconcile the plan", "what's next", or asks to move the project forward.
---

# advance-plan

Move the wakeUp project forward by one honest, verified increment. The plan doc
(`docs/PLANNING.md`) is the source of intent; the repo is the source of truth.
This skill makes them agree, then does the next small thing.

Do **not** skip the reconcile step and jump to coding — a plan that lies about
its own state is how projects drift. Reconcile first, every time.

## Step 1 — Read the plan and the repo

1. Read `docs/PLANNING.md` in full. It is organised into phases with
   `[x]` / `[ ]` checkboxes and ✅/🔨/⬜ status markers, plus a **Next actions**
   list at the bottom.
2. Read `README.md` and `Makefile` for the intended entry points.
3. Survey the actual code and tests before trusting any checkbox:
   - `src/wakeUp/` (data, attacks, features, models, eval, config, geo)
   - `tests/`, `scripts/run_milestone.py`, `configs/`
   Use Glob/Grep/Read — don't assume a box is accurate because it's ticked.

## Step 2 — Reconcile

For each phase item, decide whether the code backs up the claimed status:

- **Claimed done, is done** → leave it.
- **Claimed done, missing or partial** → downgrade the marker and note it.
- **Claimed planned (`[ ]`/⬜), actually already implemented** → upgrade it.
- **Exists in code but absent from the plan** → add a line for it.

Verify, don't guess: a feature counts as done only if the module exists AND has
a test or a working entry point. Run `python -m pytest -q` (or `make test`) when
in doubt about whether something works.

Apply the corrections directly to `docs/PLANNING.md` with Edit. Keep the exact
existing format (checkbox + trailing ✅/🔨/⬜, the phase-header status line, the
table style). If nothing is stale, say so explicitly and move on.

## Step 3 — Pick the next manageable step

Choose **one** increment — the smallest slice that leaves the repo working and
is defensible on its own. Prefer, in order:

1. The first unblocked item in **Next actions**.
2. The earliest ⬜ item in the lowest-numbered unfinished phase.

"Manageable" means: one module or one feature, shippable with a test, no
half-finished second thing. If the top Next-action is genuinely large (e.g. the
Transformer), carve off its first real sub-step (e.g. sequence tensors +
per-channel normalisation) rather than attempting the whole thing.

If the choice is ambiguous or a step is bigger than one sitting, state your pick
and your reasoning in one or two sentences, then proceed — don't stall for
approval on a routine step. Only ask the user when two paths genuinely diverge
the project.

## Step 4 — Implement it

Honor the project's design principles (top of PLANNING.md): determinism (every
injector/model takes an explicit `rng` or uses `config.set_global_seed`), the
canonical schema (`mmsi, timestamp, lat, lon, sog, cog, heading`),
source-agnostic flow, offline reproducibility, honest per-attack evaluation.

- Match the surrounding code's style, naming, and structure. Read a neighbouring
  module in the same subpackage first.
- Detectors expose `fit` / `score` where **larger score == more anomalous** so
  the eval harness stays uniform. New detectors must follow this.
- Add or extend a test in `tests/` asserting the new behaviour (labels, shapes,
  determinism as appropriate). Run the tests and report the real result — if
  they fail, say so with the output rather than papering over it.
- Wire it into the relevant `Makefile` target / `scripts/run_milestone.py` if it
  belongs in the reproducible path.

## Step 5 — Record it

Update `docs/PLANNING.md`: tick the box you completed, bump the phase status
marker if the phase advanced, and refresh **Next actions** to drop the finished
item and surface the new front-runner. Keep the representative-result and
milestone prose accurate — don't invent numbers; only cite metrics you actually
produced this run.

## Step 6 — Report

Tell the user, briefly:
- what was stale in the plan and how you corrected it,
- which step you did and why it was the right next one,
- test/verification result (real output),
- what the next step will be.

Do not commit or push unless the user asks.

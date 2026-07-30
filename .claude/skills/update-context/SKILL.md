---
name: update-context
description: Refresh the wakeUp project's written context (docs/PLANNING.md, README.md, and CLAUDE.md) so it accurately reflects the current state of the repo — after a work session, before handing off, or whenever the docs have drifted from the code. Use when the user says "update context", "update the docs", "sync the plan", "refresh the README", or "capture what changed".
---

# update-context

Keep the project's written record honest with the code. This skill only writes
documentation — it does not implement features. Its job is to make
`docs/PLANNING.md`, `README.md`, and `CLAUDE.md` describe what is *actually*
here, right now.

Truth flows from the repo to the docs, never the reverse. If a doc claims
something the code doesn't back up, the doc is wrong — fix the doc.

## Step 1 — Establish current state

1. Check what changed since the docs were last accurate:
   - `git log --oneline -15` and `git status` for recent work and uncommitted
     changes.
   - `git diff --stat HEAD~5..HEAD` (adjust range) for the shape of recent work.
2. Survey the code so claims can be verified:
   - `src/wakeUp/` (data, attacks, features, models, eval, config, geo),
     `tests/`, `scripts/`, `configs/`, `Makefile`.
   - Use Glob/Grep/Read. A capability counts as real only if the module exists
     AND has a test or a working entry point.
3. If unsure whether something works, run `python -m pytest -q` (or `make test`)
   and use the actual result — never document a passing state you didn't observe.

## Step 2 — Update the documents

Edit only what is stale; preserve each file's existing structure, tone, and
formatting conventions.

**`docs/PLANNING.md`** — the phase tracker.
- Correct every `[x]`/`[ ]` checkbox and ✅/🔨/⬜ marker to match reality
  (same rules as reconciliation: done means module + test/entry point).
- Add lines for anything built but unlisted; downgrade anything claimed but
  missing.
- Refresh the **Next actions** list: drop finished items, surface the real
  front-runner.
- Keep the representative-result and milestone prose accurate. Only cite metrics
  you produced or verified this session — do not invent or carry forward numbers
  you can't confirm.

**`README.md`** — the outside view.
- Ensure the reproduce-in-one-command instructions still work (correct script
  names, Makefile targets, install extras from `pyproject.toml`).
- Update any feature list / results summary that has drifted.

**`CLAUDE.md`** (repo root) — durable guidance for future sessions.
- Create it if absent. Keep it short and high-signal: how to run tests, build
  data, train, and eval; the canonical schema; key design invariants
  (determinism via `config.set_global_seed` + explicit `rng`; detectors expose
  `fit`/`score` with larger == more anomalous); where the main modules live.
- Record only non-obvious, stable facts — not a restatement of the file tree or
  anything git history already captures.

## Step 3 — Report

Summarise, briefly:
- which docs changed and the substantive corrections (not every word),
- anything the code does that the docs had been silent or wrong about,
- test/verification result if you ran it (real output).

Do not commit or push unless the user asks. If every doc was already accurate,
say so plainly rather than inventing edits.

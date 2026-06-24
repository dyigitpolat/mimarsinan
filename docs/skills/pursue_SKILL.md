---
name: pursue
description: Autonomously pursue a mimarsinan research/engineering goal to completion using the program's disciplined incremental method — decompose into isolated parallel workflows, tests-first + adversarial-verify + patch-for-review + byte-identical-default-off, raise honest MEASURED coverage of the deployment hypervolume, and never defer what we already understand or can fix in scope. Invoke when the user sets a goal and wants independent execution toward the toolchain-framework deliverable.
---

# Pursuing a goal in the mimarsinan program

This is the operating manual for taking a goal and driving it to a *measured, audited* DoD —
the way the program has been run. Read `docs/research/ROADMAP.md` (what gates what),
`PROGRAM_CHECKPOINT_v2.md` (state), `HYPERVOLUME.md` (the axis SSOT), and `PROGRAM_PLAN_v2.md`
before starting; activate `env/bin/activate` for anything Python.

## 1. The frame — what success means
mimarsinan is a **composable generic experimentation environment**. The deliverable is
**honest, measured, AUDITABLE coverage of the deployment hypervolume** + tools that achieve
**publication-grade results**. Energy/accuracy/speed are **per-cell outputs**, not the goal.
Every increment must do exactly one of: **(a) make the measurement trustworthy · (b) raise
honest coverage · (c) instrument a per-cell output · (d) add a hypervolume region.** If it does
none of these, it is not progress.

## 2. The method — how to make an increment
1. **Decompose** the goal into INDEPENDENT work-units with disjoint file/run ownership. Scope each
   to the smallest verifiable slice.
2. **Run each as a parallel, ISOLATED dynamic Workflow** (worktree isolation): Research → Design →
   Prototype → Verify (for new capability) or Build → Verify (for a known change). Launch
   independent units together so the fleet stays busy.
3. **Every unit is:** TESTS-FIRST · ADVERSARIALLY VERIFIED (the verifier *defaults to refuted*,
   re-derives the claims from source, mutation-checks the tests) · **DEFAULT-OFF BYTE-IDENTICAL**
   (a new capability ships gated off so the default path is provably unchanged) · returns a
   **PATCH for review** (never auto-merges framework code).
4. **Land only well-grounded, byte-identical/tested implementations.** Keep preliminary or
   not-yet-bit-exact research **ISOLATED on a branch (NOT merged)** — research is allowed to be
   incomplete; main is not.
5. **After each land:** independently re-test the branch, `safe_merge` it, **re-price the honest
   coverage / re-run the affected measurement**, update `ROADMAP.md` + the relevant design/finding
   doc, and propose the next unit. The number must be **measured, not asserted**.

## 3. Cardinal rules (never violate)
- **Base-check every isolated build first:** `git rev-parse HEAD` == current main; `reset --hard`/merge
  if stale. Agents bootstrapping on an ancestor (the `bcacfeb` trap) silently waste the run.
- **Never `git stash pop` to round-trip** — the shared stash list holds the user's work (4 stashes);
  use `git show <ref>:<path>` or a worktree. If a pop happens, recover via `git stash store` from the
  dangling commit and verify the list is intact.
- **Never resolve conflicts in the runner's live checkout.** Resolve in a throwaway worktree first, then
  `scripts/campaign/safe_merge.sh <conflict-free-branch>` (it pauses the runner, settles, clean-merges).
- **Tests-first, and never weaken an assertion / silence a check** to make code pass.
- **Commit only when the work is landed; NEVER push** unless the user explicitly asks (they control
  push + review branches).
- End commit messages with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Read + update the relevant `ARCHITECTURE.md`; keep `__init__.py` exports conservative; new dirs get
  `__init__.py` + `ARCHITECTURE.md`.

## 4. Honesty discipline (the deliverable's integrity)
- **Coverage:** the denominator is a function of **screening status** — an axis collapses *only* with a
  linked screening artifact, else it is `ASSERTED_UNSCREENED` and counted interacting. **Never merge
  VALID + VALID_FLAGGED** in any headline. Every `VALID_FLAGGED` flag carries an **owner + fix-path**;
  the report tracks **flag aging** (resolved ⇒ vision, piling up ⇒ drift). Always show the **claimed
  sub-product size** next to a fraction. Mark each region **attribution** vs **value-domain-only**.
- **Cost:** a cost number ships only with a **stated, defensible per-phase model + an uncertainty band**.
  An indefensible coefficient is worse than an empty column (it poisons the Pareto).
- **Confounds named inline** (sample noise, training-floor, crashed/non-finalized runs, instrument
  caveats). Adversarial verifiers default-refute and flag confounds rather than confirm.
- **No premature deferral:** do NOT punt to "future work" anything we already understand, or already
  understand how to fix, within this program's scope. Future-work is only *genuine* open research.

## 5. The autonomous campaign loop (keep GPUs on valuable work, never idle on waste)
scheduler FILLS the queue from `runs/campaign/backlog.json` · runner DRAINS · director GROWS the
backlog + flags `harvest_todo` · a research-round Workflow consolidates · a cron heartbeat is the
fallback. Validity (≥on-chip floor) **and** capacity (peak-phase) **pre-checks reject infeasible/invalid
configs at enqueue** so no GPU is claimed for them. **Kill-gate** quarantined/broken work (e.g.
cascaded-rescue behind the Pareto). "Never idle" means "never idle on **invalid or low-information**
work" — an idle GPU beats a row that will be retired or re-answers a settled cell.

## 6. Operating procedure per goal
1. Map the goal onto `ROADMAP.md` layers (A measurement · B breadth · C per-cell outputs · D
   capabilities · E decisions · F rigor). Identify the **highest-leverage units** + their dependencies.
2. Launch the independent units as **isolated parallel workflows** (cheap-first; no GPU-weeks until the
   cheap trustworthy layer is in). Keep a workflow always in flight; relaunch on completion.
3. On each completion: test + review the patch yourself → `safe_merge` if clean → **re-price the honest
   measurement** → update `ROADMAP.md` (status) + the design/finding doc → propose/launch the next unit.
4. **Stop on a measured DoD**, not an asserted one. Report each increment as a one-paragraph synthesis
   (what landed + the re-priced number + what's now open), not a context dump.
5. Surface to the user only genuine forks (a load-bearing methodological choice, a GPU-weeks commitment,
   a result that contradicts a prior claim). Otherwise proceed.

## 7. Final deliverables (the end of the program)
1. **The toolchain framework** — generic SNN-deployment experimentation across a wide hypervolume,
   with publication-grade tools, genericity measured + audited by construction.
2. **A research document** — the system, the research process, the **quantitative insights**, and the
   **genuine future-direction gaps** (per §4: do not defer what is already understood or fixable in scope).

# Certification Protocol — the per-cell Pareto/regression gate (Frontier E6)

**The protocol the review (C3) and the action plan (Fix B requirement #1) call "the
one you do not yet have."** Fix B turns the proven fast/lossless recipes on by default;
it **breaks every byte-identity equivalence lock by design** because it changes the
deployed numbers. This document defines the mechanism that *replaces* byte-identity as
the gate for that behavior change: a **frozen per-cell regression floor** + a **per-cell
Pareto/regression gate** on the deployed-forward metric and the wall-clock budget.

> **Mechanism only.** The code (`mimarsinan.chip_simulation.certification`), the freezing
> script (`scripts/freeze_certification_floor.py`), and this document land as *pure
> additive infra* — byte-identical, because no existing deployment path reads the floor
> yet. **Populating the floor is a GPU matrix run that happens *before the first Fix-B
> flip*, not part of this unit.**

## Why byte-identity can no longer be the gate

The V1–V8 refactor locked the pipeline against a 7,680–15,360-config byte-identity
cross-product. Those locks *enshrine the current slow/lossy default* (review C1/C3): the
instant Fix B switches a fast/lossless lever on, the deployed accuracy and wall-clock
move, and every byte-identity lock fails. You cannot ship a deliberate behavior change
through a gate whose entire job is "nothing changed." You need a gate that says **"this
commit changes the numbers, and here is the certified new baseline it must respect"** —
deployed accuracy must not regress, wall-clock must not blow the budget, **per cell**.

## The cell granularity: (firing × sync × backend)

Fix B rolls out **per cell, lowest-risk-first** (LIF → cascaded-TTFS → analytical →
synchronized), and a given (firing × sync) deploys onto a specific **backend**
(`nevresim` / `sanafe` / `lava` / `hcm`). So the floor — and the gate — is keyed by the
triple **(firing × sync × backend)**:

| field | meaning | examples |
|-------|---------|----------|
| `firing` | the spiking mode | `lif`, `rate`, `ttfs`, `ttfs_quantized`, `ttfs_cycle_based` |
| `sync` | the `ttfs_cycle_schedule` (or `None`) | `cascaded`, `synchronized`, `None` |
| `backend` | the deployment simulator measured | `nevresim`, `sanafe`, `lava`, `hcm` |

The canonical **cell key** is `mode[/schedule]@backend` (e.g.
`ttfs_cycle_based/cascaded@nevresim`, `lif@nevresim`). This reuses the `mode[/schedule]`
naming the E3 calibration keying and E4 proposer already use, so a cell names the *same
thing* across the whole program. `CertificationCell.from_mode_policy(policy,
backend=...)` builds the cell from a `SpikingModePolicy`, so the certification cell and
the policy cell are guaranteed to agree.

## The frozen-floor FORMAT

A floor book is a JSON file: `format_version` + `floors` (a map `cell_key →
RegressionFloor`). Each floor is fully self-describing — it carries the numbers AND the
tolerances the gate compares against, so the gate has no hidden global state.

```json
{
  "format_version": 1,
  "floors": {
    "lif@nevresim": {
      "deployed_accuracy": 0.9784,
      "wall_clock_s": 60.0,
      "eps": 0.0,
      "wall_clock_slack": 0.0,
      "wall_clock_budget_s": null,
      "provenance": { "commit": "deadbee", "frozen_at": "2026-...Z", "samples": 10000 }
    },
    "ttfs_cycle_based/cascaded@nevresim": {
      "deployed_accuracy": 0.95,
      "wall_clock_s": 70.0,
      "eps": 0.01,
      "wall_clock_slack": 0.5,
      "wall_clock_budget_s": null,
      "provenance": { "commit": "deadbee" }
    }
  }
}
```

| field | role |
|-------|------|
| `deployed_accuracy` | the deployed-forward, full-test, parity-gated number (R6 / E5 — the *only* number of record). The new run must not regress below `deployed_accuracy − eps`. |
| `wall_clock_s` | the measured per-step wall-clock the floor was frozen at. |
| `eps` | accuracy slack: the gate's accuracy floor is `deployed_accuracy − eps`. |
| `wall_clock_slack` | fractional speed budget: `budget = wall_clock_s × (1 + slack)`. |
| `wall_clock_budget_s` | absolute wall-clock budget override (wins over `wall_clock_slack`). |
| `provenance` | how/when frozen (commit, ISO timestamp, sample count) — the audit trail for "here is the new certified baseline." |

`format_version` is checked on load: a format change (or an unknown floor field) fails
loud instead of silently mis-reading an old book.

## The GATE

```python
from mimarsinan.chip_simulation.certification import (
    CertificationCell, certify, load_floor_book,
)

book = load_floor_book("docs/certification/regression_floor.json")
cell = CertificationCell("ttfs_cycle_based", "cascaded", "nevresim")

verdict = certify(
    cell,
    deployed_accuracy=measured_deployed_accuracy,  # the R6 number of record
    wall_clock_s=measured_wall_clock_s,
    floor_book=book,
)
assert verdict.passed, verdict.reason
```

`certify` returns a `CertificationVerdict` with one of three statuses:

* **PASS** — `deployed_accuracy >= floor − eps` **AND** `wall_clock_s <= budget`. The
  change is certified for this cell.
* **FAIL** — accuracy regressed and/or the budget was blown. `accuracy_ok` /
  `wall_clock_ok` decompose *which* side failed; `reason` names it.
* **MISSING_FLOOR** — no frozen floor for this cell. It **cannot** be certified — a
  missing floor is **never a silent pass**. Freeze the floor first.

## How to FREEZE (before the first Fix-B flip)

1. **Run the matrix** (a GPU run — *not* part of E6) at the CURRENT slow/lossy defaults
   to get, per (firing × sync × backend) cell, the deployed-forward full-test accuracy
   (behind the R6 torch↔sim parity gate) and the per-step wall-clock.
2. **Freeze each cell** with the script (one invocation per cell — a reviewable diff):

   ```bash
   python scripts/freeze_certification_floor.py \
       --book docs/certification/regression_floor.json \
       --firing ttfs_cycle_based --sync cascaded --backend nevresim \
       --deployed-accuracy 0.95 --wall-clock-s 70.0 \
       --eps 0.01 --wall-clock-slack 0.5 \
       --commit "$(git rev-parse --short HEAD)" --samples 10000
   ```

   The script reads the existing book (or starts a new one), records the cell
   immutably, and writes the JSON with stable key order (clean diff). It **does not run
   the matrix** — you supply the already-measured numbers.
3. **Commit the frozen book** as the certified baseline. This is the "here are the
   numbers Fix B must respect" artifact.

## How to GATE (each Fix-B flip)

Roll out **per cell, lowest-risk-first**, never a blanket flip:

1. Flip the cell's recipe on (LIF first — the safest).
2. Re-measure that cell's deployed-forward accuracy + wall-clock (R6 gates on).
3. `certify(cell, ...)` against the frozen book. **PASS → ship the flip and re-freeze
   the cell at its new (better) numbers as the new baseline; FAIL → do not ship.**
4. Only after a cell certifies, proceed to the next (cascaded → analytical →
   synchronized).

This converts AC4 ("no regression") from a hope into a runtime invariant, and gives the
clean "this commit changes numbers, here is the new certified baseline" story the review
asks for. See `docs/FRONTIER_PROGRAM.md` § "The transition protocol (C3)".

## What E6 is NOT

* It does **not** run the matrix or populate any floor (that GPU run precedes the first
  flip).
* It does **not** flip any Fix-B lever or change any default (those are Fix B, gated by
  this protocol).
* It is **byte-identical**: no existing deployment path reads the floor; the mechanism
  is dormant until Fix B wires `certify` into the deployment gate.

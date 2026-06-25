# Finding: the enqueue capacity gate let ~10% of campaign runs burn ~20 min of GPU, then crash at mapping

**Status:** SOLVED + landed (test-first, validated on the real corpus).
**Axis:** measurement trust / campaign efficiency (`/pursue §1a`, `§5` "never idle on invalid work").

## The issue (confirmed)

126 of 1267 campaign run logs (~10%) died with
`RuntimeError("No more hard cores available")` — each **after ~20 min of GPU
training**, then crashing at the Hard Core Mapping step. ~40 GPU-hours wasted on
jobs that were doomed before they started.

The enqueue **capacity pre-check existed** (`scheduler.capacity_precheck` →
`estimate_cores_needed`) but admitted these jobs. Root cause: `estimate_cores_needed`
is a **SOUND LOWER bound** — it sums axons/neurons GLOBALLY across a segment and so
ignores that the greedy packer keeps each **threshold group on its own hard cores**.
`threshold_group_id` is the softcore's **perceptron index** (`legacy_convert.py:127`,
`canonical._read_threshold_group`) — a *structural*, weight-independent property. The
bound mixes groups the packer cannot; on `deep_cnn` d6/d8 it under-counts:

| config (deep_cnn d8 KMNIST sync) | lower bound | budget | real packer |
|---|---|---|---|
| `neural_segment_until:features_13` | 252 total | 280 | overflows mid-pack (>280) |

The feasibility boundary is multi-dimensional — the **same** `(depth,width,count)`
appears in both done and failed jobs; the swing factor is the **dataset input
resolution** (CIFAR 32×32×3 vs MNIST 28×28×1 feature maps) and the core count — so no
clean analytic patch to the bound is sound across all axes.

## The fix (study-a-solution, landed)

`mapping/verification/capacity/dryrun.py` — `dryrun_pack_feasible(ir_graph, pc)`:
**run the actual hybrid packer** on the (weight-independent) IR and return a definitive
`PackFeasibility`. Wired into `capacity_precheck` after the lower bound admits
(`capacity_dryrun_gate`, default-on; opt-out `=False`; failures NON-FATAL so a builder
edge case never drops valid work). Costs **~1 s of CPU at enqueue** (the IR build the
estimate already does) vs. ~20 min of GPU.

Because the threshold grouping is **structural**, the untrained dry-run is bit-identical
to the trained deployment's packing — an **exact feasibility oracle, zero false-rejection**.

## Validation (real corpus, no mocks)

- **CRASH configs: 60/60 sampled (of 126) → REJECT** at enqueue, each naming the
  overflowing segment. 100% recall on doomed jobs.
- **DONE configs: 80/80 sampled (of 801) → ADMIT.** Zero false-rejections of jobs that
  actually deployed.
- The default `run.py` pipeline is **unchanged** — `dryrun.py` is consumed only by the
  campaign scheduler's enqueue gate.

## Repro

```python
from mimarsinan.mapping.verification.capacity import dryrun_pack_feasible, estimate_cores_needed
# 8 cores, distinct perceptron indices (= threshold groups), budget 4:
#   lower bound = ceil(16/8) = 2 ≤ 4  → ADMITS (fooled)
#   real packer = one core per group  → 8 > 4 → REJECTS
```

`tests/unit/mapping/test_capacity_dryrun.py` (the gap distilled to a deterministic unit),
`tests/unit/gpu/test_scheduler_capacity_gate.py` (the wired gate: reject pack-infeasible,
admit pack-feasible, skip on lower-bound rejection / gate-off, non-fatal on error).

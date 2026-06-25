# D4 — Structured pruning → fewer cores → fewer reprogram phases → lower cost

**Status:** MEASURED demo + default-off deployment wiring landed.
**Unit:** `wave9/pruning-deploy-cost`.

## Claim

Structured magnitude pruning, applied to a model **before mapping**, structurally
removes whole output neurons (and the matching downstream axons), so the deployed
model maps to **fewer hard cores**, needs **fewer reprogram phases** under
scheduling, reloads **fewer weights**, and therefore lands a **lower
weight-reuse cost band**. This is the D4 cost lever the A2 pruning screen consumes.

All numbers below are **MEASURED** by the production instruments — never asserted —
and are locked by `tests/unit/transformations/test_d4_pruning_scheduling_cost_demo.py`.

## How it is wired (default-off, byte-identical)

A new opt-in deployment parameter `prune_sparsity` (default `0.0`, unset ⇒ no
pruning) is resolved on `DeploymentPlan` and applied by `SoftCoreMappingStep`
**before the IR is built**, via
`mimarsinan.transformations.pruning.magnitude.prune_perceptron_chain`
(`soft_core_structured_pruning.apply_structured_pruning_if_enabled`).

* `prune_sparsity` unset / `0.0` ⇒ the hook is a **no-op**: the SAME `nn.Linear`
  objects with bit-exact tensors, and an **identical** mapped core count. The
  default deployment / `run.py` path is provably unchanged.
* `prune_sparsity > 0` ⇒ the lowest-magnitude `prune_sparsity` fraction of each
  perceptron's output channels is structurally dropped (logits and the network
  input are exempt), shrinking `out_features` and the downstream `in_features`.

The hook runs at SCM time — i.e. **after** `NormalizationFusionStep` has folded
each perceptron's normalization into its `nn.Linear` (`normalization == Identity`)
— so shrinking the linear's output rows is structurally sound (no stale affine-norm
vector is left to mismatch the pruned width).

## Measured demo

**Model.** A real perceptron-flow MLP in its post-fusion (Identity-norm) SCM-time
state, input `1×16×16`, widths `[256, 256, 256, 256, 10]` (3 wide hidden layers +
the 10-logit head). Wide intermediate widths so the **diagonal core bound** —
`max(⌈Σaxons/max_axons⌉, ⌈Σneurons/max_neurons⌉, frags·groups)` — not a fixed
floor, governs the mapped core count, so the structural shrink visibly drops it.

**Instruments (consumed, not modified).**
`IRMapping.map` → `estimate_cores_needed` (static cores + scheduled `phase_count`)
→ `weight_reuse_plan_from_graph` (reprogram passes + params reloaded)
→ `phase_cost_band` (the cited low/nominal/high weight-reuse cost band, default
coefficient band: E_dma 31/160/320 pJ/byte, bytes/param 0.5/1.0/2.0,
E_sync 0.1/1/10 µJ/barrier).

**Budgets.** Single-pool core budget `max_axons=256, max_neurons=64,
allow_coalescing=True` for the static core count; a tight **scheduled** budget
(`count=8`, `allow_scheduling=True`) so oversized segments need multiple reprogram
passes and pruning provably cuts `phase_count`.

### Dense vs pruned (measured)

| metric | dense | pruned `s=0.25` | pruned `s=0.50` |
|---|---|---|---|
| intermediate widths | 256 / 256 / 256 | 192 / 192 / 192 | 128 / 128 / 128 |
| **hard cores** (`estimate_cores_needed`) | **13** | **10** | **7** |
| **reprogram phases** (`weight_reuse_plan_from_graph`) | **13** | **10** | **7** |
| params reloaded | 199,168 | 124,800 | 66,816 |
| **scheduled `phase_count`** (tight budget) | **2** | **2** | **1** |
| cost band low / nominal / high (mJ) | 4.387e-3 / **4.487e-2** / 2.575e-1 | 2.934e-3 / 2.997e-2 / 1.799e-1 | 1.736e-3 / **1.769e-2** / 1.128e-1 |

**Headline.** At `s=0.50` the deployment maps to **7 cores instead of 13**
(-46%), reloads **66,816 weights instead of 199,168** (-66%), drops from **2 to 1**
reprogram passes under the tight scheduled budget, and the nominal weight-reuse
cost band falls from **4.49e-2 mJ to 1.77e-2 mJ — a 2.54× reduction**. Every band
endpoint (low / nominal / high) strictly drops, so the saving holds across the
entire cited uncertainty range.

## Why structured, not masked

The in-loop mask-and-rescale pruner (`transformations.pruning.apply`) zeros/down-
scales weights but keeps every channel **resident**, so the crossbar shapes — and
thus the mapped core count — are unchanged. Only **structured** removal of whole
output neurons + matching downstream axons drops the per-segment diagonal bound,
which is the quantity `estimate_cores_needed` sums. That is why D4 wires the
structured `prune_perceptron_chain` transform at the pre-mapping seam.

## Reproduce

```
PYTHONPATH=src:../spikingjelly env/bin/python -m pytest \
  tests/unit/transformations/test_d4_pruning_scheduling_cost_demo.py \
  tests/unit/pipelining/pipeline_steps/test_soft_core_structured_pruning.py -q
```

The demo test locks the exact integer measurements above and the ~2.5× nominal
savings factor; the wiring test locks the plan resolution, the pruned-vs-dense
core/phase/cost reduction, and the default-off byte-identical no-op.

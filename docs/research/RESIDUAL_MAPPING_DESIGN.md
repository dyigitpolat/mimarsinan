# Residual-connection mapping design (`y = x + F(x)`)

Status: design synthesis (2026-06-25). Author: residual-mapping ARCHITECT.

This document specifies the **simplest correct mechanism** for deploying a
residual connection `y = x + F(x)` through the mimarsinan pipeline, grounded in
a live audit of the topology, latency, placement, and forward-semantics seams.

## TL;DR (the headline)

A residual ADD is **already fully representable and deployable today**, bit-exact,
with **zero code changes**. It lands as a multi-input host-side `ComputeOp`
(`ComputeAdapter(operator.add)`), exactly the path `SkipPerceptronMixer` uses
(`skip_perceptron_mixer.py:123,144`) and the path the converter auto-emits for any
FX `+` node (locked by `tests/unit/torch_mapping/test_unified_converter.py::TestAddRoutesThroughGenericPath`).

The design therefore has two tiers:

- **Tier 0 (ship now, host-side merge):** use the existing `ComputeOp` add. No new
  code. This is the recommended first prototype and the correct fallback.
- **Tier 1 (follow-up, on-chip merge):** lower a *param-free* add to an on-chip
  signed-IF `NeuralCore` with a frozen identity-concatenation weight bank `[I; I]`,
  so the sum happens on a crossbar and the on-chip fraction stays high. This is the
  one genuinely open seam; it is **not** needed for a correct first deployment.

The five required decisions are answered below. The honest feasibility verdict
(section 6) is: **Tier 0 is one prototype pass and already green; Tier 1 is a
separate, scoped follow-up round.**

---

## 1. IR representation of `y = x + F(x)`

**Decision: a multi-input `ComputeOp` merge node — NOT a new node type, NOT (in Tier 0) an on-chip additive core.**

The topology is already multi-predecessor capable at all three layers:

- **FX trace:** `x + F(x)` records as `call_function operator.add` (or
  `call_method add`/`__add__`), each with 2 `fx.Node` args.
  `RepresentabilityAnalyzer` classifies every `call_function`/`call_method` as
  supported (`representability_analyzer.py:145-164`); add is never rejected.
- **Mapper DAG:** `ComputeOpMapper` holds `_sources_list` of `1..N` sources and
  branches on `len(sources)`: `_emit_unary` vs `_emit_multi`
  (`compute_op_mapper.py:124-155, 220-238`). The diamond (one producer fanning to
  two consumers) is two mappers referencing the **same** source-mapper object.
- **IRGraph:** `IRNode.input_sources` is an arbitrary array of
  `IRSource(node_id, index)` (`ir/types.py:27-47`). `_emit_multi` concatenates
  both branches' source arrays via `concat_source_views` and emits **one**
  `ComputeOp` whose `input_sources` spans both predecessors
  (`compute_op_mapper.py:229`, `ir_mapping_class_base.py:119`).

**Authoring a residual programmatically** (matches `skip_perceptron_mixer.py:123`):

```python
out = ComputeOpMapper([F_branch_mapper, skip_branch_mapper],
                      ComputeAdapter(operator.add))
```

The converter does this automatically from an FX `+` node via
`_emit_generic_compute_op` (`mapper_graph_fx.py:66-89`), which partitions args into
mapper-sources vs const-tensors and builds a `ComputeAdapter(operator.add)` with
`_bound_count == 0` (param-free → zero bound tensors → stays picklable).

**Invariant:** a bare add requires broadcast-compatible branch widths.
`ComputeOpMapper._check_broadcastable` (`compute_op_mapper.py:135-148`) raises
`ShapeMismatchError` when skip-width != F-width. A **dimension-changing residual**
(projection skip) is therefore NOT a bare add: insert a projection
`PerceptronMapper` on the skip branch first, then add.

---

## 2. Latency alignment (how the skip path meets F(x) at the merge)

**Decision: NO delay buffer and NO shiftable-core reuse is needed in Tier 0. Alignment is by topological execution order + state-buffer refcounting, not by cycle timing.**

This is the most important correction to the naive hypothesis. The two inputs to
the add do **not** live in the cycle-accurate domain, so there is no "merge cycle"
to align to and no missing delay to insert.

- Each `ComputeOp` is partitioned into its **own** `compute` stage
  (`HybridStage.kind == "compute"`), separate from `neural` stages.
- Stages run **strictly sequentially** (`run_hybrid_stages`,
  `hybrid_stage_runner.py:38`). Each neural segment is decoded to **rates**
  (`decode_segment_output_torch = counts / T`, `segment_boundary.py:57-66`) before
  the next stage runs.
- The skip producer's rate persists in `state_buffer[node_id]` across **all**
  intervening F-path stages until the add gathers both inputs. A refcount
  (`decref_consumers`, `hybrid_execution.py:222-239`) keeps a value alive until its
  last consumer reads it.

So the skip rate is held in the buffer for "free" — the buffer **is** the delay
line, indexed by IR `node_id`, not by cycle. `ChipLatency` / `_align_shiftable_cores`
(`latency/chip.py:114-143`) operates only **within a single neural segment** on
cycle indices; it is a pruning-artifact fixup, irrelevant to a cross-segment
host merge. Per `mapping/LATENCY.md`: IRLatency = topology tier; ChipLatency =
per-segment cycle index. **Do not** try to bend ChipLatency to delay a skip.

**Tier 1 caveat:** if and only if the merge is moved on-chip into the same
cycle-accurate segment as F's last layer, then the standard `ChipLatency`
invariant `consumer.lat >= max(live source lat) + 1` and the existing
`_align_shiftable_cores` pass already handle it — the skip producer core becomes
a normal cross-core source and is aligned automatically, because LIF re-emits each
cycle in `[src_lat, src_lat+T)`. No new latency mechanism is required even for
Tier 1; the existing per-segment alignment generalizes.

---

## 3. Placement

**Tier 0 decision: host-side `ComputeOp` (param-free `ComputeAdapter(operator.add)`).**

- Deploys via `hybrid_execution.py:_compute_op` / `execute_compute_op_torch`
  (`:74-98`): gather both predecessor buffers, run `operator.add`, store result.
- Consumes **zero chip cores** (good for the area budget) but **breaks pure-on-chip
  deployment** and incurs a host/chip round-trip at the merge. Acceptable for a
  first deployment and as the permanent fallback.

**Tier 1 decision (preferred long-term, open seam): on-chip signed-IF merge core.**

Lower a param-free `operator.add` to `IRMapping.add_neural_core`
(`ir_mapping_class_emit.py:33`) instead of a host `ComputeOp`:

- **Weights:** a frozen, param-free identity-concatenation bank
  `W = [I_f ; I_f]` — `axons = 2*f`, `neurons = f`, no bias. The two branches are
  concatenated into one axon space (`concat_source_views` already produces exactly
  this spanning `input_sources`) and matmul by `[I; I]` sums them on the crossbar.
- **Signed-IF:** the merge must be a **signed** integrate-and-fire core (no ReLU
  rectification of the sum), per the LIF signed-IF fix
  (`lif_signed_integrate_and_fire`). Set `activation_type` accordingly so the
  membrane charges signed.
- **Topology is already legal:** a `NeuralCore` may take `input_sources` spanning
  both branches; `IRSource`/`input_sources` already permit it.

What's missing for Tier 1 is **only** a mapper (or a branch in `_emit_multi`) that
*chooses* the on-chip-core lowering over the host-ComputeAdapter lowering for
param-free elementwise add. The topology, the emit API, and the latency alignment
all already support it.

> Do **not** mistake `flowchart_node_estimate` (`compute_op_mapper.py:92-122`) for an
> existing on-chip lowering. It only *describes* add as a linear op (`concat +
> identity weights`) for visualization; it emits nothing.

---

## 4. Forward semantics in `SpikingHybridCoreFlow`

**Decision (Tier 0): the add is a host compute stage that operates on decoded
rates; correctness is preserved per-mode by the existing scale plumbing.**

In `_forward_rate` (`rate_forward.py:107-133`, `_on_compute_rate`):

1. The state buffer holds **rates in `[0, 1]`** (LIF) at compute time.
2. `resolve_stage_compute_scales(..., apply_ttfs=False)` → the rate/LIF path does
   **not** apply TTFS scales; the add runs directly on rates.
3. `execute_compute_op_torch` gathers both branch rates and computes their sum.
4. `encode_compute_boundary` re-encodes the summed result back into the spike
   domain for the next neural segment (`rate_forward.py:123-133`).

**Scale correctness across the two summed branches** (the real numerical risk):
the two branches may live in different absolute rate domains. `ComputeOpMapper`
already carries `per_source_scales` / `output_scale` (`compute_op_mapper.py:61,
157-168`), and at emission it wraps the module in `ScaleNormalizingWrapper`
(`compute_modules.py:72-117`), computing
`f(r_1·s_1, ..., r_N·s_N) / s_out` so the add happens in **absolute** units while
inputs/outputs travel as rates. This is the mechanism that keeps the sum
numerically correct in rate/TTFS deployment; it is populated by the scale-propagation
policy (`scale_propagation.apply_compute_op_scale_policy`).

**Bit-exact NF↔SCM preservation:** the host `ComputeOp` is identical numpy/torch
code in both the NF (torch hybrid) and SCM paths
(`execute_compute_op_torch` ⇄ `execute_compute_op_numpy`,
`hybrid_execution.py:178-205`, with `dtype=float64` for HCM parity). There is no
spike-timing seam in Tier 0 — the add is pure host arithmetic — so NF↔SCM parity
is structurally exact and is what the existing `SkipPerceptronMixer` already relies
on.

**Tier 1 forward semantics:** an on-chip signed-IF merge sums spike/value streams
on the crossbar within one cycle-accurate window. The per-branch absolute scales
that `ScaleNormalizingWrapper` applies host-side must instead be **baked into the
identity weights** (`W = [s_1·I ; s_2·I] / s_out`) so the on-chip sum matches the
host reference. Validate NF↔SCM↔HCM per-neuron via the existing
`nf_scm_parity` / `assert_torch_sim_fidelity` harness before trusting it.

---

## 5. Minimal prototype plan

**Smallest residual model:** one residual block `y = x + Linear(x)` (equal width,
so the add is a bare elementwise add — no projection). Concretely a 3-layer MLP:
`stem (Linear) -> z`; `F = Linear(z)`; `y = z + F`; `head (Linear) -> logits`.
This is structurally the `SkipPerceptronMixer` block stripped to its essence and is
small enough to run NF↔SCM↔HCM in seconds.

**Smallest code change to deploy bit-exact: ZERO.** Tier 0 needs no production code
change. The deliverable is a **test + a tiny model fixture**, following the CLAUDE.md
test-first discipline:

1. Add a `MinimalResidualBlock(nn.Module)` test fixture (equal-width skip).
2. **Test A — representability + IR shape:** assert `convert_torch_model` succeeds,
   the IR contains exactly one `ComputeOp` with `ComputeAdapter(operator.add)`,
   `_bound_count == 0`, and `input_sources` spanning both branches with distinct
   internal predecessors (the diamond). (Mirror
   `TestAddRoutesThroughGenericPath`.)
3. **Test B — bit-exact forward:** assert `max |y_torch - y_sim| == 0.0` (float64)
   for the NF torch hybrid forward, and NF↔SCM↔HCM agreement via the existing
   `assert_torch_sim_fidelity` lock (identity-mapping rung). LIF mode first
   (`lossless`-capable per MEMORY); it is the safest mode to claim bit-exact.
4. **Test C — dimension-changing residual rejection:** assert a width-mismatched
   bare add raises `ShapeMismatchError`, documenting the projection-skip
   requirement.

**Tier 1 prototype (separate follow-up round):**

1. Add `add_identity_merge_core` mapper (or a `_emit_multi` branch gated on
   `is_unbound_add`) that emits `add_neural_core` with a frozen `[I; I]` bank, no
   bias, signed-IF `activation_type`.
2. Bake per-branch scales into the bank (`[s_1·I ; s_2·I] / s_out`).
3. Lock NF↔SCM↔HCM per-neuron parity for the on-chip merge with the
   `nf_scm_parity` gate before flipping any default.
4. Keep it **config-gated default-off** (Boy-Scout + the repo's graduation
   discipline): host-ComputeOp stays the default until the on-chip merge is
   parity-proven across modes.

---

## 6. Feasibility — honest verdict

| Item | One prototype pass? | Notes |
|------|---------------------|-------|
| **Tier 0**: host-side residual add, bit-exact | **YES — already green** | Zero production code; the path is locked by existing tests and used by `SkipPerceptronMixer`. Prototype = a fixture + 3 parity tests. |
| Equal-width residual representability | YES | Bare `ComputeAdapter(operator.add)`, multi-input ComputeOp. |
| Cross-segment latency alignment | YES (no work) | Buffer refcounting holds the skip rate; no delay buffer, no shiftable-core change. |
| Per-branch scale correctness | YES | `ScaleNormalizingWrapper` already handles it; populated by the scale-propagation policy. |
| NF↔SCM bit-exactness | YES | Host arithmetic shared by both paths; structurally exact (LIF first). |
| Dimension-changing (projection) residual | Needs a projection core first | Not a bare add; insert `PerceptronMapper` on the skip branch. Still Tier-0-deployable once projected. |
| **Tier 1**: on-chip signed-IF identity-merge core | **NO — separate round** | New mapper + scale-baking into `[I;I]` + full NF↔SCM↔HCM parity lock + config gate. Topology/emit/latency already support it; only the lowering choice is missing. |

**Bottom line:** ship Tier 0 now (it is effectively already shipped and merely
needs explicit test coverage on a minimal residual block), and scope Tier 1
(on-chip param-free additive core) as a focused follow-up that reuses every
existing seam — `add_neural_core`, signed-IF, `concat_source_views`,
`ChipLatency` per-segment alignment, and the `nf_scm_parity` gate.

## Key files / seams

- `src/mimarsinan/torch_mapping/mapper_graph_converter.py` — FX add dispatch (`:158`, `:232`).
- `src/mimarsinan/torch_mapping/mapper_graph_fx.py` — `_emit_generic_compute_op` (`:66`).
- `src/mimarsinan/mapping/mappers/compute_op_mapper.py` — `_emit_multi` (`:220`), scale wrap (`:157`), flowchart estimate (visualization-only, `:92`).
- `src/mimarsinan/mapping/support/compute_modules.py` — `ComputeAdapter` (`:17`), `ScaleNormalizingWrapper` (`:72`).
- `src/mimarsinan/mapping/ir/types.py` — `IRSource`/`ComputeOp`/`NeuralCore` (`:27`, `:81`, `:192`).
- `src/mimarsinan/mapping/ir_mapping_class_base.py` — `add_compute_op` emission seam (`:119`).
- `src/mimarsinan/mapping/ir_mapping_class_emit.py` — `add_neural_core` (Tier 1 on-chip seam, `:33`).
- `src/mimarsinan/mapping/latency/chip.py` — `_align_shiftable_cores` (per-segment only, `:114`).
- `src/mimarsinan/chip_simulation/hybrid_run/hybrid_execution.py` — host ComputeOp exec (`:74`), refcount (`:222`).
- `src/mimarsinan/models/spiking/hybrid/rate_forward.py` — `_on_compute_rate` forward semantics (`:107`).
- `src/mimarsinan/spiking/segment_boundary.py` — decode `/T` (`:57`), compute-boundary re-encode.
- `src/mimarsinan/models/perceptron_mixer/skip_perceptron_mixer.py` — existing residual (`:123`,`:144`).
- `tests/unit/torch_mapping/test_unified_converter.py::TestAddRoutesThroughGenericPath` — the lock.

## §7. Tier-1 status (round 2 — mechanism landed on branch, NOT bit-exact; not merged)

Round 2 (branch `residual-tier1-onchip`, commit `bed8b36`, **isolated/not merged**) built the
on-chip param-free identity-merge core: a **mapper-graph rewrite** (`mapping/support/residual_merge.py::lower_residual_adds_to_onchip_merge`)
replaces the param-free host `ComputeAdapter(operator.add)` with a frozen identity-concat `[I|I]`
signed-IF merge Perceptron (no bias, requires_grad=False), fed by a `_ResidualConcatMapper` that
concats both branches into one 2·width axon space — reusing the existing
`PerceptronMapper→map_fc→add_neural_core` path (so neuron_split / axon_fuse / coalescing work for
free). Config-gated `onchip_residual_merge` (default OFF → byte-identical host add).

**Proven:** param-free (count_host_params unchanged), on-chip fraction ON≥OFF, the merge stays in
ONE neural segment (no host barrier), the residual sum is real, default-OFF byte-identical, 906+
mapping/spiking tests unbroken.

**OPEN (round 3):** full NF==HCM **bit-exactness is not reached** (max|Δ|=0.125; per-neuron parity
0.25 — honestly xfailed). The in-segment merge diamond has two sources at **different cascade
depths** (stem d1, F d2); `LifSegmentPolicy.run_segment`'s per-cycle cascade does not reproduce the
HCM per-source latency-**window** integration at the merge boundary (stale/zero source buffer past
`src_lat+T`), leaving a sparse ~1/T residual. A naive depth-aware train-shift over-corrected
(off-by-1 → off-by-4) and was reverted. **Round 3 = a focused, depth-aware per-source window
alignment in `LifSegmentPolicy.run_segment` matching the HCM merge-window boundary**, then wire
`lower_residual_adds_to_onchip_merge` into the production conversion flow behind the flag.

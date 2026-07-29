# Structured Row/Column Elimination for Quantized IMC Inference — Engineering Handoff

**Audience:** the research team building the experimental apparatus and running the study.
**Scope:** non-spiking (`core_semantics="mvm"`) deployment, swept weight quantization, no AQ, no SNN.
**Status:** the *method* is implemented and in-tree. The *measurement and sweep apparatus* is the work.

---

## 1. Executive summary

The headline mechanism — structured row/column elimination with a bidirectional cross-layer
propagative cascade — **already exists in this codebase and is exercised by the test suite.** The
work ahead is not to build the cascade; it is to *measure* it, *ablate* it, and *sweep* it.

Three things were built during this handoff (§5). Six remain (§6). The single most important
remaining item is **W1, the cascade equivalence certificate** — until it exists, no accuracy number
from this study can be attributed between "the criterion cost us accuracy" (a result) and
"compaction/rewiring has a bug" (a defect).

---

## 2. The architecture, in one law

```
     SEED                     CASCADE                      SETTLE
     criterion                fixpoint (DCE)               resource accounting
     torch model masks   ->   IR graph liveness fixpoint -> crossbars + occupancy
     LOSSY                    SEMANTICS-PRESERVING         MEASUREMENT
```

> **Governing law: elimination is dead-code elimination on the mapping graph. The criterion is the
> only lossy step.**

Everything follows from this. The cascade removes structure that provably cannot influence the
output, so it must be **bit-exact**; if it is, every accuracy delta is attributable to the seeds
alone, and the resource-accuracy trade-off curve is honest by construction.

### Why the IR, not the torch model

The cascade belongs in the IR graph because the IR encodes producer/consumer structure at
*axon and neuron granularity* — precisely the granularity of crossbar rows and columns. A torch
`Linear` feeds an anonymous tensor; an IR `NeuralCore` feeds typed `IRSource` edges that name
`(source_node, source_column)`. That is what makes "this neuron's death removes that axon row in
every consumer" expressible at all.

### The exactness precondition (currently unchecked — see W1)

A column with no live source still emits `act(0 + bias)`. Elimination is exact only when
`bias == 0` **and** `act(0) == 0`. True for ReLU/GELU/identity; **false for sigmoid** and shifted
activations. Under `mvm` the activation is a host ComputeOp, so this must become a checked
predicate rather than an assumption. The `cols_with_implicit_source` parameter already exists as
the escape hatch — a column with a live out-of-matrix source is never killed by propagation.

---

## 3. Verified inventory — what already exists

Every row below was read during the investigation; the pointers are exact.

### 3.1 The cascade

| Component | Location |
|---|---|
| Cascade driver | `mapping/pruning/ir_pruning_core.py::prune_ir_graph` |
| Intra-matrix fixpoint | `mapping/pruning/graph/pruning_propagation.py::compute_propagated_pruned_rows_cols` |
| Global cross-node fixpoint | `mapping/pruning/graph/pruning_graph_core.py::compute_global_pruned_sets` |
| Node liveness | `mapping/pruning/ir_liveness.py` — `NodeLiveness.{LIVE,BIAS_ONLY,DEAD}` |
| Bias-activation semantics | `mapping/pruning/liveness_semantics.py::bias_can_activate` |
| Compaction (physical shrink) | `mapping/pruning/ir_pruning_compact.py::_compact_node` (`np.ix_`) |
| Edge rewiring after removal | `mapping/pruning/ir_pruning_helpers.py::_rewire_sources` |
| Model I/O exemptions | `mapping/pruning/boundary_policy.py` |
| Deployed neuron survival | `mapping/pruning/deployed_neuron_survival.py` |
| Pipeline call site | `pipelining/pipeline_steps/mapping/soft_core_mapping_ir_pruning.py::apply_ir_pruning_if_enabled` |

`prune_ir_graph`'s own docstring states the contract:

> *"Pruning is bidirectional/recursive across NeuralCore boundaries; ComputeOps block functional
> propagation. Model input data axons and output logits are never pruned; DEAD cores are deleted,
> surviving cores compacted."*

**`NodeLiveness.BIAS_ONLY` is the "layer collapsed to a 1×1 bias neuron" phenomenon** observed
previously — it is a named, handled state, not an accident.

**Non-spiking liveness is already correct.** A hypothesis that the liveness predicate was
spiking-coupled and would over-prune under `mvm` was **refuted by reading the code**:
`bias_can_activate` branches on `INERT_SPIKING_MODE` with *"Value domain (mvm): no threshold, no
window — any nonzero bias is live."* **Do not "fix" this.**

### 3.2 The criterion (structured, already row/column-wise)

| Component | Location |
|---|---|
| Row/col group saliency | `transformations/pruning/masks.py` — `row_l1 = weight.abs().sum(dim=1)`, `col_l1 = ...sum(dim=0)` |
| Structured channel pruning | `transformations/pruning/magnitude.py::prune_perceptron_chain` |
| Activation-based importance | `transformations/pruning/activation.py` |
| Mask commit/verify at cache seams | `transformations/pruning/committed_masks.py` |
| Seeds extracted for the IR | `mapping/pruning/ir_pruning_masks.py::get_initial_pruning_masks_from_model` |

### 3.3 Both pruning regimes exist (answers "one-shot vs recover")

| Regime | Entry point | Config gate |
|---|---|---|
| **One-shot structured, pre-mapping** | `pipeline_steps/mapping/soft_core_structured_pruning.py::apply_structured_pruning_if_enabled` | `prune_sparsity > 0` |
| **Progressive prune→recover** | `pipeline_steps/adaptation/pruning_adaptation_step.py::PruningAdaptationStep` (a `TunerPipelineStep`, so it trains) | `pruning: true` + `pruning_fraction` |

### 3.4 Platform geometry (already generic)

`platform_constraints["cores"]` is a **list of heterogeneous core types**
`{max_axons, max_neurons, count, has_bias}` — an arbitrary mixed-tile IMC chip is already
expressible. Two consumers read it with *different and both-correct* semantics:

- `mapping/platform/platform_constraints.py::resolve_platform_mapping_params` — takes the **max**
  across core types to bound how large a single softcore tile may be.
- the packer (`mapping/packing/hybrid_build_scheduled.py`, `cores_config`) — honours **per-type
  populations** verbatim when placing.

Permission bits live separately in `mapping/platform/mapping_structure.py::ChipCapabilities`
(`allow_coalescing`, `allow_neuron_splitting`, `allow_scheduling`, `schedule_policy`, …).

**A platform is therefore: (heterogeneous core-type list) × (capability bits) × (weight width).**

### 3.5 Deployment + certification substrate (landed previously)

`core_semantics="mvm"` (`chip_simulation/core_semantics.py`), `PackagingContract`
(`mapping/platform/packaging_contract.py`), the value executor
(`chip_simulation/value_run/`), and the R-edge/C-edge value twin certificates. 5% pruning on
`mvm` tier-0 rows is green with FATAL certs — pruning × mvm × weight banks is already exercised.

---

## 4. Confirmed gaps

| ID | Gap | Severity |
|---|---|---|
| **G-F** | Cascade is **uncertified**. Function preservation is asserted, never proven. | **Keystone** |
| **G-A** | Utilization metrics exist **only on the SNN path** (`chip_simulation/sanafe/`). | Critical *(closed, §5)* |
| **G-B** | No attribution: cannot separate "criterion killed it" from "propagation killed it". | Critical |
| **G-C** | No cascade-off ablation → the single-layer baseline is unbuildable. | Critical *(seam closed, §5)* |
| **G-G** | Quantization-induced deadness not harvested: low-bit WQ creates new exact zeros that should re-seed the cascade. | High |
| **G-H** | ComputeOps block propagation **uniformly** — caps cascade reach on transformers/residual nets. | High |
| **G-D** | No named platform registry. | Medium *(closed, §5)* |
| **G-E** | No sweep/harvest driver. | Medium |

---

## 5. Implemented during this handoff

All landed with tests; full suite **8,759 passed**, typecheck **0 errors**.

### 5.1 `mapping/crossbar_utilization.py` — the results table (closes G-A)

`CrossbarUtilizationReport`, mirroring the existing `WeightProgrammingReport` in shape and intent.

- `CoreOccupancy.from_hard_core(core)` — reads `HardCore`: used = `axons_per_core − available_axons`.
- `CrossbarUtilizationReport.from_hard_cores(...)` / `.from_hybrid_mapping(...)` — the latter walks
  every neural stage's `hard_core_mapping.cores` across the whole program.
- Aggregates: `cores_allocated`, `axon_utilization`, `neuron_utilization`, `cell_occupancy`,
  `unusable_space`, `macs`, `programming_bits` (= cells × weight bits; `None` when the platform
  declares no width).
- `.to_dict()` returns a **flat scalar record** — one row of the experiment table, harvest-ready.

### 5.2 `mapping/platform/imc_platforms.py` — named geometries (closes G-D)

`IMCPlatform(name, cores, weight_bits, provenance, capabilities)` + `register_imc_platform` /
`get_imc_platform` / `imc_platform_names`, with `.to_platform_constraints()` rendering the body the
pipeline already consumes.

**Heterogeneous tiles are first-class and tested end-to-end:** `heterogeneous_platform()` builds
mixed-tile chips from `(max_axons, max_neurons, count)` triples, tile types stay *separate* through
`to_platform_constraints()`, non-square crossbars are expressible, and a test drives a
heterogeneous platform into the real `resolve_platform_mapping_params`. `validate()` rejects
missing/zero geometry loudly at registration.

> **Registered geometries are PLACEHOLDERS** (`PLACEHOLDER_PROVENANCE`). Sourcing real per-chip
> numbers is a literature pass. `provenance` must name a citation before any registered platform is
> used for a published measurement. `imc_mixed_tile` exists to exercise the heterogeneous path.

### 5.3 Propagation ablation seam (closes G-C at the kernel)

`compute_propagated_pruned_rows_cols(..., propagate: bool = True)`. With `propagate=False` the
fixpoint is skipped and exactly the seeded, exemption-filtered sets are returned — **the
single-layer structured-pruning baseline: identical criterion, identical rate, no cascade.**
Tests pin that the baseline is a strict subset of the cascade and that exemptions still hold.

*Remaining wiring (mechanical):* thread `propagate` from a config key through
`prune_ir_graph` → `compute_global_pruned_sets` → the kernel. See W3.

---

## 6. Work remaining

### W1 — Cascade equivalence certificate ★ **do first**

**Why first:** every number downstream is uninterpretable without it.

- New `mapping/pruning/cascade_certificate.py`.
- Certify `value(pruned+compacted IR) == value(seeded-but-uncompacted IR)` **bit-exact** over N
  batches. Reuse `chip_simulation/value_run/ValueHybridCoreFlow` and
  `mapping/packing/hybrid_build_pool.build_identity_hybrid_mapping` — the same machinery the R/C-edge
  certs use. Pattern to copy: `tests/unit/chip_simulation/test_value_flow_memory.py`.
- Add a `ZeroPreservingActivation` predicate (§2). A column whose consumer activation fails
  `act(0) == 0` routes into `cols_with_implicit_source` instead of dying. **Unknown ops fail loud.**
- Verify the shared-bank union rule: a bank column may die only if dead for **all** instances that
  share the bank (`initial_pruned_per_bank`, `_attach_bank_metadata`).
- **DoD:** FATAL cert green on one conv + one transformer vehicle; a deliberately corrupted
  `_rewire_sources` must trip it (mutation test).

### W2 — Emit the utilization report on every run

`CrossbarUtilizationReport` exists but nothing calls it yet. Wire it where
`WeightProgrammingReport` is emitted (`mapping/weight_programming.py`, printed + reporter-evented on
every Hard Core Mapping run), serialize `.to_dict()` to the run dir as JSON, and add a ratchet test
that an `mvm` run emits it. **DoD:** every mvm run drops a utilization record.

### W3 — Elimination ledger + ablation wiring

- `EliminationLedger`: rows/cols killed by **seeds** vs **propagation** vs **liveness-DEAD**, per
  node and per bank; count of cores deleted; count of `BIAS_ONLY` collapses.
  `_attach_pre_compaction_metadata` and `store_heatmap` already capture pre-compaction state — build
  on those rather than re-walking.
- Thread `propagate` (§5.3) through `prune_ir_graph` → `compute_global_pruned_sets` from a config key
  (suggest `elimination_propagation`, default `true`).
- **DoD:** hand-built graph where propagation provably kills more; `propagate=false` reproduces the
  seed set exactly; ledger totals reconcile against the compacted shapes.

### W4 — Post-quantization cascade re-run (G-G)

Low-bit WQ creates **new exact zeros**, which are new seeds. This is the paper's
"interaction with low-bit quantization".

- **First, audit the step order** — does IR pruning run before or after weight quantization?
  *Pre-registered falsifier:* if pruning already runs after WQ, W4 collapses to a no-op and the
  ledger will show zero quantization-induced eliminations. Check that before building anything.
- Re-run the fixpoint on the **deployed (quantized)** matrices.
- **DoD:** a matrix whose small weights vanish at 4b yields strictly more eliminations than at 8b.

### W4b — ComputeOp liveness transfer functions (G-H) ★ **highest research value**

Today ComputeOps block propagation *uniformly* — safe, but it confines the cascade inside blocks and
bounds the transformer result. Replace the uniform barrier with a dataflow framework: a per-op
`LivenessTransfer` registry with a **conservative opaque default**.

| ComputeOp | Transfer semantics |
|---|---|
| elementwise add (residual) | channel-aligned join: output channel dead iff dead in **all** branches; a dead output propagates backward into every branch |
| concat | disjoint index ranges — exact bijection, propagates both directions |
| reshape / permute / transpose | index permutation — propagate through the index map |
| activation (zero-preserving) | identity on liveness |
| layernorm / softmax / attention | mixes across the reduced axis — **opaque** |
| unknown | **opaque** (default; never guesses) |

Same shape as a compiler's dataflow framework: registered transfer functions, safe default, monotone
fixpoint (liveness only shrinks ⇒ termination is free). This is what turns residual/transformer nets
from "cascade stops at the block boundary" into whole-model propagation.
Start at `mapping/pruning/boundary_policy.py::_computeop_relays_deadness`, which is the existing
per-op relay predicate and the natural seam to generalize.
**DoD:** a residual chain propagates across the add; an attention block still blocks; every transfer
function is covered by the W1 certificate.

### W5 — Real platform specs

Replace `PLACEHOLDER_PROVENANCE` with sourced geometries + citations. Design work is done; this is a
literature pass. Use the `asta-papers` tooling; **never hand-edit `.bib`**.

### W6 — Sweep + harvest driver

`scripts/sweep/structured_elimination_sweep.py`: cartesian
(vehicle × platform × weight_bits × sparsity × regime × propagation) → configs → runs → tidy
CSV/JSON of accuracy + `utilization.to_dict()` + ledger. Rides the closed row schema
(`templates/generate.py::validate_row_keys`) so a typo'd key fails loud instead of evaporating.

### W7 — Docs + ratchets

`ARCHITECTURE.md` for `mapping/pruning` and `mapping/platform`; ratchet that the utilization report
is emitted on every mvm run. (`mapping/ARCHITECTURE.md` already updated for
`crossbar_utilization.py`.)

**Order:** W1 → W2/W3 → W4 → W4b → W5 → W6 → W7.

---

## 7. The two pruning regimes — trade-offs

Both exist (§3.3); the study should report both, and they answer different questions.

| | **One-shot structured** (`prune_sparsity`) | **Progressive prune→recover** (`pruning` + `pruning_fraction`) |
|---|---|---|
| Cost | One mapping pass; minutes | Retraining in the loop; ~10× or more |
| Accuracy at high sparsity | Degrades sharply | Substantially better — the network re-allocates capacity |
| What it measures | The *structure* the criterion + cascade expose, uncontaminated by training | The deployable operating point |
| Confound | None — weights are fixed | Recovery can re-grow importance into rows the criterion killed, changing what cascades |
| Use for | Sweeping the resource axis densely; ablating propagation; the quantization interaction | The headline accuracy-vs-resources Pareto points |

**Recommended protocol:** sweep *one-shot* densely across
(platform × weight_bits × sparsity × propagation on/off) to map the resource surface cheaply and get
the propagation-vs-single-layer comparison at many rates; then run *progressive* at a handful of
selected sparsities to establish the deployable accuracy points. This keeps the expensive regime
proportional to the claims that need it.

One interaction to watch: the recover loop can revive a row the criterion had zeroed, which *undoes*
a cascade that had already propagated. Whether the ledger is recomputed after each recovery round is
a design decision the team must make explicitly — record it, because it changes what "final
sparsity" means.

---

## 8. Running an experiment

Config essentials for this study:

```jsonc
{
  "core_semantics": "mvm",          // non-spiking; no AQ machinery required
  "weight_quantization": true,
  "weight_bits": 4,                  // the swept axis
  "activation_quantization": false,  // out of scope
  "prune_sparsity": 0.5,             // one-shot structured
  // or: "pruning": true, "pruning_fraction": 0.5   // progressive + recover
  "platform_constraints": { /* IMCPlatform.to_platform_constraints() */ }
}
```

- Run from the project root with `run.py`; **always activate `env` first.**
- Gates before any commit: `python -m pytest tests` (≤2 min, parallel, always green) and
  `./scripts/typecheck.sh` (zero errors).
- Template/tier rows: `templates/generate.py` is the SSOT — edit and regenerate, never hand-edit the
  JSONs. Their tests run separately: `pytest scripts/template_tests`.
- `mvm` supports **torch-category models only** — native builders raise by design
  (`DeploymentPlan.resolve`).

---

## 9. Risks and open questions

1. **Cascade reach on transformers is unmeasured.** Run the cheap falsifier *before* the expensive
   sweep: measure reach on one conv and one ViT vehicle at fixed sparsity. If ComputeOp blocking
   confines the ViT cascade within blocks, W4b must move ahead of W6 and the experiment matrix
   should re-weight toward conv workloads until it lands.
2. **Shared weight banks bound elimination.** A bank column can only die if dead for *all* instances
   sharing it, so weight-shared (transformer FC, scheduled) layers will cascade less than
   owned-weight layers. This is a genuine result to report, not a bug — but it must be *measured*,
   because it means "sparsity" is not comparable across sharing regimes.
3. **`act(0) == 0` is currently assumed.** Until W1 lands, any vehicle with a non-zero-preserving
   activation is a silent-correctness risk.
4. **Placeholder geometries must not reach a paper.** Enforced socially by `provenance`; consider a
   test that fails if a run cites a platform still carrying `PLACEHOLDER_PROVENANCE`.

---

## 10. Discipline (from `CLAUDE.md`, non-negotiable)

- **Tests first** — they dictate design. Run them after every change.
- **Never weaken an assertion or silence a check** to make code pass.
- **Fail loud.** The only sanctioned log-and-degrade seam is `common.best_effort` and only for
  telemetry/rendering — never for verification, mapping, or training logic.
- Ratchet tests only tighten. Never grow an allowlist without a stated contract reason.
- Update the touched module's `ARCHITECTURE.md` when adding/renaming a direct child — the drift
  meta-test enforces it (it caught `crossbar_utilization.py` during this handoff).
- Commit messages: plain and conventional, **no AI-attribution lines**.

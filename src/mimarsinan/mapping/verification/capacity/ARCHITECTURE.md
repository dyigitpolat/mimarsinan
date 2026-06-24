# mapping/verification/capacity/ — Static Placement-Capacity Diagnostic (E4)

A pure, fast, SOUND lower bound on the hard cores an `IRGraph` needs on a given
core budget — computed WITHOUT running the greedy placer — so a provably-infeasible
config is rejected EARLY with a diagnosable verdict instead of crashing late inside
the packer with `"No more hard cores available"` (E3: VGG16@224 needs ~316k cores
on a 1000-core budget).

## Key modules

| File | Symbols | Role |
|------|---------|------|
| `estimate.py` | `estimate_cores_needed`, `CapacityEstimate`, `CapacityExceededError` | Partitions the IR into neural segments (`layout.segmentation.partition_ir_graph`); each segment's bound mirrors the diagonal packer = `max(ceil(Σ axons / max_axons), ceil(Σ neurons / max_neurons), max per-core frags·groups)` (`platform.coalescing.coalescing_fragment_count` × `ceil(out / max_neurons)`); sums across segments (one core pool) and names the first cumulative overflow. The budget is `sum(count)` over `cores`; `effective_max_axons` follows `resolve_platform_mapping_params` (loses one to the bias row when no `has_bias`). Because it is a LOWER bound it NEVER rejects a config the packer could place (VGG16@32 → 830 ≤ 2048; lenet5 → 40 ≤ 120, actual placed 57; deep_cnn d8 → 224 ≤ 368 suggested). `CapacityEstimate.raise_if_infeasible()` raises `CapacityExceededError(cores_needed, cores_available, overflowing_segment)` — the gate. |

## Exported API (`__init__.py`)

`estimate_cores_needed`, `CapacityEstimate`, `CapacityExceededError` (re-exported by the parent `mapping.verification` package).

## Dependencies

- **Internal**: `mapping.layout.segmentation` (`partition_ir_graph`, `NeuralSegment`), `mapping.platform.coalescing` (`coalescing_fragment_count`), `mapping.platform.platform_constraints` (`resolve_platform_mapping_params`).

## Dependents

- `SoftCoreMappingStep._run_capacity_gate` — the early capacity gate before HardCoreMapping (config `capacity_gate` default-on).
- `scripts/campaign/scheduler.py` `capacity_precheck` — the enqueue-time capacity gate (rejects infeasible configs before they claim a GPU; estimate failures NON-FATAL).

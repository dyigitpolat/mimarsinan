# mapping/verification/capacity/ — Static Placement-Capacity Diagnostic (E4)

Two layered feasibility checks on an `IRGraph` against a core budget, both run
WITHOUT a GPU/sim so a doomed config is rejected EARLY (before claiming a GPU)
instead of crashing late inside the packer with `"No more hard cores available"`:

1. **SOUND lower bound** (`estimate.py`) — a pure, fast bound that rejects only
   configs PROVABLY too big for any packing strategy (E3: VGG16@224 needs ~316k
   cores on a 1000-core budget). Being a lower bound, it never false-rejects but is
   LOOSE: it ignores threshold-group fragmentation.
2. **Real-packer DRY-RUN** (`dryrun.py`) — runs the ACTUAL hybrid packer to catch
   what the lower bound misses. The greedy packer keeps each threshold group (a
   softcore's PERCEPTRON INDEX — structural, weight-independent) on its own hard
   cores, so a config can pass the bound yet exhaust the budget mid-pack (`deep_cnn`
   d6/d8 on high-resolution inputs: lower bound 252 ≤ 280, real pack overflows at
   `features_13`). Because the grouping is structural, the untrained dry-run matches
   the trained deployment's packing exactly — an EXACT oracle, no false-rejection.

## Key modules

| File | Symbols | Role |
|------|---------|------|
| `estimate.py` | `estimate_cores_needed`, `CapacityEstimate`, `CapacityExceededError` | Partitions the IR into neural segments (`layout.segmentation.partition_ir_graph`); each segment's bound mirrors the diagonal packer = `max(ceil(Σ axons / max_axons), ceil(Σ neurons / max_neurons), max per-core frags·groups)` (`platform.coalescing.coalescing_fragment_count` × `ceil(out / max_neurons)`); sums across segments (one core pool) and names the first cumulative overflow. The budget is `sum(count)` over `cores`; `effective_max_axons` follows `resolve_platform_mapping_params` (loses one to the bias row when no `has_bias`). Because it is a LOWER bound it NEVER rejects a config the packer could place — but it is LOOSE: it sums axons/neurons GLOBALLY, ignoring that the packer keeps each threshold group on its own cores (deep_cnn d8 → 252 ≤ 280 admits, real pack needs >280 → use `dryrun.py`). `CapacityEstimate.raise_if_infeasible()` raises `CapacityExceededError(cores_needed, cores_available, overflowing_segment)` — the gate. **Scheduling-aware (E4 r2):** `estimate_cores_needed(..., allow_scheduling=None)` resolves the scheduling permission from the chip's `MappingStrategy` capability (`ChipCapabilities.from_platform_constraints`; explicit arg wins). When ON, the SCHEDULED path reprograms a FRESH core pool per phase, so feasibility is decided by the PEAK per-segment phase fitting the budget (`peak_phase_cores = min(max segment bound, budget)`) PLUS a hard atomic-unit gate (the largest single coalescing bundle `frags·groups` of one softcore cannot split across phases, so it must fit the WHOLE budget); `phase_count = Σ ceil(segment_bound / budget)` (reprogram-pass count). NO weight sharing assumed (a 224²-conv genuinely unrolls; scheduling only time-multiplexes). `CapacityEstimate` carries `scheduled`/`peak_phase_cores`/`phase_count` (default `False`/`cores_needed`/`1` ⇒ `allow_scheduling=False` is byte-identical to the SUM verdict). VGG16@224 on 256×256×2048 reads feasible-via-scheduling, peak 2048, ~158 phases. |
| `dryrun.py` | `dryrun_pack_feasible`, `PackFeasibility` | Runs the REAL hybrid packer (`packing.hybrid_hardcore_mapping.build_hybrid_hard_core_mapping`) on the IR — the definitive feasibility verdict the lower bound cannot give. Resolves the same `MappingStrategy` the mapping step uses (so a scheduling-feasible config is admitted), catches the packer's `RuntimeError("No more hard cores available")`, and parses the overflowing segment from the diagnostic; returns `PackFeasibility(feasible, hard_cores, overflowing_segment, error)`. Only the capacity-exhaustion `RuntimeError` is caught — structural errors (e.g. empty-graph `ValueError`) propagate so an upstream NON-FATAL wrapper admits rather than mislabeling them infeasible. Threshold groups are perceptron-indexed (structural, weight-independent), so an untrained dry-run is bit-identical to the trained deploy's packing → EXACT oracle (validated: 60/60 crash configs rejected, 80/80 done configs admitted). |

## Exported API (`__init__.py`)

`estimate_cores_needed`, `CapacityEstimate`, `CapacityExceededError`, `dryrun_pack_feasible`, `PackFeasibility` (re-exported by the parent `mapping.verification` package).

## Dependencies

- **Internal**: `mapping.layout.segmentation` (`partition_ir_graph`, `NeuralSegment`), `mapping.platform.coalescing` (`coalescing_fragment_count`), `mapping.platform.platform_constraints` (`resolve_platform_mapping_params`), `mapping.platform.mapping_structure` (`ChipCapabilities`/`MappingStrategy` — SSOT for the `allow_scheduling` permission bit + resolved packing strategy), `mapping.packing.hybrid_hardcore_mapping` (`build_hybrid_hard_core_mapping` — the real packer the dry-run invokes).

## Dependents

- `SoftCoreMappingStep._run_capacity_gate` — the early capacity gate before HardCoreMapping (config `capacity_gate` default-on).
- `scripts/campaign/scheduler.py` `capacity_precheck` — the enqueue-time capacity gate (rejects infeasible configs before they claim a GPU). Runs the SOUND lower bound, then the real-packer DRY-RUN (`capacity_dryrun_gate` default-on) to catch threshold-group-fragmentation overflows the bound misses. Estimate/dry-run failures NON-FATAL.

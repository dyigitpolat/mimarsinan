# mapping/packing/ — Softcore and Hardcore Packing

Maps IR neural segments to `SoftCore` / `HardCore` layouts and hybrid programs.

## Key modules

| File | Symbols | Role |
|------|---------|------|
| `spike_source.py` | `is_off`, `is_input`, `is_always_on`, `source_is_*` | Shared spike-source sentinel predicates for compaction and remap |
| `canonical.py` | `canonical_*`, `pick_best_softcore` | Shared feasibility, split, fuse protocols for layout + runtime packers |
| `greedy/` | `greedy_pack_softcores`, split helpers | Greedy bin-packing main loop and neuron-split helpers |
| `core_packing.py` | Re-exports | Public packing API |
| `placement_engine.py` | `Materializer`, `run_placement` | Single greedy placement engine. The `Materializer` strategy (layout shape-only vs runtime weight-bearing) supplies the place / fuse / split hooks; both `pack_layout` and `HardCoreMapping.map` drive the identical assignment kernel through it. |
| `softcore/` | `SoftCore`, `HardCore`, `HardCoreMapping`, `RuntimeMaterializer`, `compact_soft_core_mapping` | Runtime soft→hard mapping split across `soft_core`, `hard_core`, `hard_core_mapping` (now a thin wrapper over `placement_engine` via `RuntimeMaterializer`), `compaction`. `HardCoreMapping.map_identity(softcore_mapping)` = 1:1 `SoftCore`→`HardCore` (no pack/pad/reindex); shares `_finalize_sources` with `map()` |
| `hybrid_types.py` | `HybridHardCoreMapping`, `HybridStage` | Hybrid program datatypes |
| `hybrid_segment.py` | `_flush_neural_segment` (`identity: bool`), … | Segment flush and IO remap |
| `hybrid_build_pool.py` / `hybrid_build_scheduled.py` | `build_hybrid_hard_core_mapping`, `build_identity_hybrid_mapping`, `_build_single_pool`, `_build_scheduled` | IRGraph → hybrid program compiler; both packed paths partition via `layout.segmentation.partition_ir_graph` (single segmentation source). The packing permissions (coalesce/split/schedule) are governed by a resolved `platform.MappingStrategy` (param `strategy=`); when omitted it is derived from the legacy `allow_*` flags, so the dispatch reads `strategy.allow_*`, never the raw flags. `build_identity_hybrid_mapping(*, ir_graph)` (re-exported via `hybrid_hardcore_mapping.py`) = 1:1 `NeuralCore`→`HardCore`, no pool/pad/reindex/coalesce/split (SCM rung-2 identity gate) |
| `neural_segment_packing.py` | `neural_segment_to_soft_core_mapping` | Neural-only segment → SCM |
| `softcore/soft_core_mapper.py` | Mapper helpers | Soft-core emission from mapper graph |

## Dependencies

- **Internal**: `mapping.ir`, `mapping.platform.mapping_structure`, `mapping.pruning.ir_segmentation`
- **External**: `numpy`

## Dependents

- `pipelining.pipeline_steps.mapping`, `models.hybrid_core_flow`, `visualization.graphviz`, SANA-FE / simulation runners.

## Invariants

- Layout and runtime packers share one engine (`placement_engine.run_placement`) and the same `canonical_*` predicates, so wizard and deployment paths agree on placement by construction (pinned by `tests/integration/test_placement_parity.py`).
- Deployment mapping statistics are derived through `layout.LayoutPlan.from_hybrid_mapping` → the shared `build_stats_from_packing_result`, the same engine the wizard miniview uses (no parallel stats formulas).
- `HardCoreMapping.merge_softcore_into` records per-softcore provenance on each placement dict: `ir_node_id` (== the source IR `NeuralCore.id`; compaction preserves ids and the segment remap rewrites only sources), `perceptron_index`, `perceptron_output_slice`, `coalescing_role`, and split metadata. This lets per-neuron NF↔HCM spike-count parity reconstruct each perceptron's neurons by concatenating its accumulator cores in IR-id order (filtering coalescing partials, reassembling neuron-split fragments by `neuron_range_in_original`). (`psum_role` is still recorded but always `None`: the lossy firing partial-sum decomposition was removed — a wide fan-in fuses into one wider crossbar instead.)

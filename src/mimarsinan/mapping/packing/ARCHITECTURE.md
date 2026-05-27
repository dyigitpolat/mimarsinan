# mapping/packing/ — Softcore and Hardcore Packing

Maps IR neural segments to `SoftCore` / `HardCore` layouts and hybrid programs.

## Key modules

| File | Symbols | Role |
|------|---------|------|
| `canonical.py` | `canonical_*`, `pick_best_softcore` | Shared feasibility, split, fuse protocols for layout + runtime packers |
| `greedy_split.py` | `_try_split_into_*` | Neuron-split helpers |
| `greedy_pack.py` | `greedy_pack_softcores` | Greedy bin-packing main loop |
| `greedy.py` | Re-exports | Compatibility aggregate |
| `core_packing.py` | Re-exports | Public packing API |
| `softcore_mapping.py` | `SoftCore`, `HardCore`, `HardCoreMapping` | Runtime soft→hard mapping |
| `hybrid_types.py` | `HybridHardCoreMapping`, `HybridStage` | Hybrid program datatypes |
| `hybrid_segment.py` | `_flush_neural_segment`, … | Segment flush and IO remap |
| `hybrid_build.py` | `build_hybrid_hard_core_mapping` | IRGraph → hybrid program compiler |
| `neural_segment_packing.py` | `neural_segment_to_soft_core_mapping` | Neural-only segment → SCM |
| `soft_core_mapper.py` | Mapper helpers | Soft-core emission from mapper graph |

## Dependencies

- **Internal**: `mapping.ir`, `mapping.platform.mapping_structure`, `mapping.pruning.ir_segmentation`
- **External**: `numpy`

## Dependents

- `pipelining.pipeline_steps.mapping`, `models.hybrid_core_flow`, `visualization.graphviz`, SANA-FE / simulation runners.

## Invariants

- Layout and runtime packers must use the same `canonical_*` predicates so wizard and deployment paths agree on placement.

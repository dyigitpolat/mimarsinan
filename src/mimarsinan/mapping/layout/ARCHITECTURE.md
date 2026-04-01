# mapping/layout/ -- Layout Estimation for Architecture Search

Provides lightweight, shape-only layout estimation used during architecture
search to evaluate hardware feasibility without constructing full models or
weight matrices.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `layout_types.py` | `LayoutSoftCoreSpec`, `LayoutHardCoreType`, `LayoutHardCoreInstance`, `LayoutCoreSnapshot`, `LayoutPackingResult` | Data classes for layout-only core specifications and packing results. `LayoutHardCoreInstance.softcore_count` tracks placements; `LayoutHardCoreInstance.unusable_space` accumulates strip-shaped packing inefficiency; `LayoutPackingResult` includes `unused_area_total`, `unusable_space_total`, and `avg_unusable_space_per_core`; `LayoutCoreSnapshot` captures per-used-core axon/neuron usage for stats; `LayoutPackingResult.used_core_snapshots` holds one snapshot per used core; `coalesced_fragment_count` and `split_fragment_count` count packing feature usage. |
| `layout_ir_mapping.py` | `LayoutIRMapping` | Collects `LayoutSoftCoreSpec`s from mapper graph traversal (shape only, no weights). Supports `allow_coalescing` and `hardware_bias` flags for axon tiling: wide FC layers emit psum-decomposed or coalescing-tiled softcores matching `IRMapping` core counts. Structural decisions (bias counting, wide-layer detection, psum params) delegated to shared `mapping_structure` helpers. **Pruning estimation** (`collect_layout_softcores`): applies 80% of pruning fraction with pessimistic heuristics — output-layer columns protected, per-bank uniform reduction, per-bank threshold group assignment. |
| `layout_packer.py` | `pack_layout` | Packs layout softcores into layout hardcores using `greedy_pack_softcores`; successful result includes `used_core_softcore_counts` and `used_core_snapshots`. |

## Dependencies

- **Internal**: `mapping.ir` (`IRSource`), `mapping.mapping_structure` (`compute_core_input_count`, `compute_fc_tiling_mode`, `compute_psum_params`), `mapping.core_packing` (`greedy_pack_softcores`).
- **External**: `numpy`.

## Dependents

- `search.problems.joint_arch_hw_problem` uses `LayoutIRMapping` and `pack_layout`
  for hardware feasibility evaluation during architecture search.

## Exported API (\_\_init\_\_.py)

All layout types (including `LayoutCoreSnapshot`), `LayoutIRMapping`, and `pack_layout`.

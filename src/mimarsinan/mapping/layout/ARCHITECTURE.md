# mapping/layout/ -- Layout Estimation for Architecture Search

Provides lightweight, shape-only layout estimation used during architecture
search to evaluate hardware feasibility without constructing full models or
weight matrices.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `layout_types.py` | `LayoutSoftCoreSpec`, `LayoutHardCoreType`, `LayoutHardCoreInstance`, `LayoutPackingResult` | Data classes for layout-only core specifications and packing results |
| `layout_ir_mapping.py` | `LayoutIRMapping` | Collects `LayoutSoftCoreSpec`s from mapper graph traversal (shape only, no weights) |
| `layout_packer.py` | `pack_layout` | Packs layout softcores into layout hardcores using `greedy_pack_softcores` |

## Dependencies

- **Internal**: `mapping.ir` (`IRSource`), `mapping.core_packing` (`greedy_pack_softcores`).
- **External**: `numpy`.

## Dependents

- `search.problems.joint_arch_hw_problem` uses `LayoutIRMapping` and `pack_layout`
  for hardware feasibility evaluation during architecture search.

## Exported API (\_\_init\_\_.py)

All layout types, `LayoutIRMapping`, and `pack_layout`.

# mapping/platform/ — Platform Constraints and Structure

Hardware constraint resolution and structural helpers shared by layout and mapping.

## Key modules

| File | Symbols | Role |
|------|---------|------|
| `platform_constraints.py` | `resolve_platform_mapping_params`, `resolve_scalar_mapping_params` | Constraint resolution from config |
| `mapping_structure.py` | `compute_fc_tiling_mode`, `compute_core_input_count`, `WideFanInUnsupportedError` | FC tiling (wide → coalescing fuse, or raise when `allow_coalescing=False`) and bias-axon counting |
| `coalescing.py` | Coalescing helpers | Wide-layer coalescing rules |

## Dependents

- `mapping.layout`, `mapping.ir_mapping`, architecture search, wizard.

# mapping/platform/ — Platform Constraints and Structure

Hardware constraint resolution and structural helpers shared by layout and mapping.

## Key modules

| File | Symbols | Role |
|------|---------|------|
| `platform_constraints.py` | `resolve_platform_mapping_params`, `resolve_scalar_mapping_params` | Constraint resolution from config |
| `mapping_structure.py` | `ChipCapabilities`, `MappingStrategy`, `compute_fc_tiling_mode`, `compute_core_input_count`, `WideFanInUnsupportedError` | Declared chip permissions + core grid (`ChipCapabilities`) and the resolver that **derives** the per-layer mapping decision — coalesce/split/sync-point — from shape × capabilities (`MappingStrategy`, extending `compute_fc_tiling_mode`); plus bias-axon counting. Capabilities/strategy untangle the three independent `allow_coalescing`/`allow_neuron_splitting`/`allow_scheduling` flags: declared once, resolved once, read as `strategy.allow_*` / `strategy.tiling_mode(...)`. `ChipCapabilities.from_platform_constraints(dict)` is the SSOT for reading the three permission bits out of a platform-constraints / wizard body dict; `permission_kwargs()` (on both `ChipCapabilities` and `MappingStrategy`) spreads them into the layout/verify helper signatures (`verify_hardware_config` / `compute_mapping_stats` / `build_layout_plan`), so entry points assemble the bits once instead of re-reading config at each call. |
| `coalescing.py` | Coalescing helpers | Wide-layer coalescing rules |

## Dependents

- `mapping.layout`, `mapping.ir_mapping`, architecture search, wizard.

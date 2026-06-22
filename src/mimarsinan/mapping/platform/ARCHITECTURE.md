# mapping/platform/ — Platform Constraints and Structure

Hardware constraint resolution and structural helpers shared by layout and mapping.

## Key modules

| File | Symbols | Role |
|------|---------|------|
| `platform_constraints.py` | `resolve_platform_mapping_params`, `resolve_scalar_mapping_params` | Constraint resolution from config |
| `mapping_structure.py` | `ChipCapabilities`, `MappingStrategy`, `compute_fc_tiling_mode`, `compute_core_input_count`, `WideFanInUnsupportedError` | Declared chip permissions + core grid (`ChipCapabilities`) and the resolver that **derives** the per-layer mapping decision — coalesce/split/sync-point — from shape × capabilities (`MappingStrategy`, extending `compute_fc_tiling_mode`); plus bias-axon counting. Capabilities/strategy untangle the three independent `allow_coalescing`/`allow_neuron_splitting`/`allow_scheduling` flags: declared once, resolved once, read as `strategy.allow_*` / `strategy.tiling_mode(...)`. `ChipCapabilities.from_platform_constraints(dict)` is the SSOT for reading the three permission bits out of a platform-constraints / wizard body dict; `MappingStrategy.from_permissions(allow_coalescing=…, allow_neuron_splitting=…, allow_scheduling=…)` wraps loose raw bools into a strategy in one call (the SSOT replacement for the removed `build_hybrid_hard_core_mapping` back-compat kwargs); `permission_kwargs()` (on both `ChipCapabilities` and `MappingStrategy`) spreads them into the layout/verify helper signatures (`verify_hardware_config` / `compute_mapping_stats` / `build_layout_plan`), so entry points assemble the bits once instead of re-reading config at each call. `allow_per_layer_s` is the **EW1 RESERVED** temporal capability gate (each cascade depth / latency group may run at its own resolution `S_d` instead of one global `simulation_steps`): declared on `ChipCapabilities` alongside `allow_coalescing`, read via `strategy.allow_per_layer_s`, but intentionally NOT in `permission_kwargs()` (it is a temporal, not a layout/verify, input) and no mapping decision consults it yet — the per-depth S map is derived by the ConversionPolicy keystone (research). Default False ⇒ uniform global S only ⇒ byte-identical. |
| `coalescing.py` | Coalescing helpers | Wide-layer coalescing rules |

## Dependents

- `mapping.layout`, `mapping.ir_mapping`, architecture search, wizard.

# visualization/graphviz/ — Mapping Graphviz DOT Writers

Generates Graphviz DOT (and optional SVG via `dot`) for IR and hardware mappings.

## Key modules

| File | Symbols | Role |
|------|---------|------|
| `common.py` | `try_render_dot`, `_dot_html_label`, helpers | Shared DOT/HTML helpers and rendering |
| `ir.py` | `write_ir_graph_dot`, `write_ir_graph_summary_dot` | IR graph diagrams |
| `softcore.py` | `write_softcore_mapping_dot` | SoftCore mapping |
| `hardcore.py` | `write_hardcore_mapping_dot` | HardCore mapping |
| `hybrid.py` | `write_hybrid_hardcore_mapping_dots`, `HybridVizArtifacts` | Hybrid program + segments |

## Dependencies

- **Internal**: `mapping.ir`, `mapping.packing.softcore_mapping`, `mapping.packing.hybrid_hardcore_mapping`
- **External**: system `dot` binary (optional)

## Dependents

- `pipelining.pipeline_steps.mapping` (`run_optional_viz`), GUI snapshots.

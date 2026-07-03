# mapping/mappers/ — Mapper Hierarchy

`Mapper` subclasses that emit IR from the `ModelRepresentation` graph.

## Key modules

| File | Role |
|------|------|
| `base.py` | `Mapper` base class + `resolve_activation_type`; `require_source_mapper()` (non-None source accessor); default V6 per-node polymorphic methods (`propagate_source_scale`, `propagate_boundary_scale`, `flowchart_node_estimate`) |
| `perceptron_mapper.py` | FC / perceptron layers (`PerceptronMapper`) |
| `compute_op_mapper.py` | Host-side ComputeOp mapper (`ComputeOpMapper`, `ShapeMismatchError`) |
| `conv1d_mapper.py` / `conv2d_mapper.py` | Shared-weight conv perceptron mappers |
| `conv_helpers.py` | Shared conv helpers: `_chunk_sizes`, `pad_source_grid` (OFF-source grid padding) |
| `module_mapper.py` | Forward-only `ModuleMapper` (identity in IR) |
| `structural.py` | Input, Reshape, EinopsRearrange, Stack, Concat, Subscript, Permute |
| `leading_dim.py` | Merge / Split / Ensure-2D leading-dimension reshapes |
| `scale_propagation.py` | V6: `walk_out_scales` (single graph walk) + shared per-kind scale-decision bodies for `compute_per_source_scales` / `propagate_boundary_input_scales` |
| `flowchart.py` | V6: `FlowchartNodeEstimate` / `FlowchartFCSpec` value objects for the softcore flowchart estimate |

## V6 — per-node polymorphism (no isinstance-on-mapper-kind chains)

Per-node graph-walk decisions live on the `Mapper` subclasses as polymorphic
methods, so a new mapper kind overrides ONE method instead of being added to
chains in 3+ files:

- `propagate_source_scale(deps, out_scales)` — weight-quant per-source scales
  (consumed by `mapping/support/per_source_scales.py`).
- `propagate_boundary_scale(deps, out_scales, default)` — TTFS theta-out boundary
  scales (consumed by `spiking/scale_aware_boundaries.py`).
- `flowchart_node_estimate(out_shape)` — software summary + optional FC estimate
  spec (consumed by `visualization/softcore_flowchart_dot.py`; the viz layer turns
  the `FlowchartFCSpec` into core counts so mappers stay free of viz deps).

The base provides transparent-routing defaults; `InputMapper`, the perceptron /
conv mappers, `ConcatMapper`, `ComputeOpMapper`, and `StackMapper` override.

## Dependents

- `mapping.mapping_utils`, `mapping.support.per_source_scales`,
  `spiking.scale_aware_boundaries`, `visualization.softcore_flowchart_dot`,
  `torch_mapping`, model builders.

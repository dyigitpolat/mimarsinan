# mapping/pruning/ — IR Pruning and Liveness

Graph pruning, liveness analysis, and segmentation metadata for hybrid mapping.

## Key modules

| File | Symbols | Role |
|------|---------|------|
| `boundary_policy.py` | `PruningBoundaryPolicy`, `compute_model_io_boundary_policy` | Model I/O hard-exempt policy (single source of truth) |
| `ir_pruning_core.py` | `prune_ir_graph` | Main prune pass |
| `ir_liveness.py` | Liveness helpers | Row/col liveness for pruning |
| `liveness_semantics.py` | TTFS/liveness rules | Activation semantics for prune decisions |
| `graph/pruning_propagation.py` | `compute_propagated_pruned_rows_cols` | Propagate prune masks |
| `graph/pruning_graph_core.py` | `compute_global_pruned_sets` | Global prune sets |
| `pruning_apply.py` | Apply helpers | Apply prune results to structures |
| `ir_segmentation.py` | `build_ir_consumed_by`, `get_neural_segments` | Segment boundary metadata (packing only) |

## Boundary policy

Pruning protects **model I/O only** (not segment staging I/O):

- **Hard exempt rows**: axons fed by `IRSource(node_id=-2)` (model input data).
- **Hard exempt cols**: neurons listed in unified `ir_graph.output_sources`.
- **ComputeOp**: relays structural deadness (upstream prune → downstream axon) without hard-exempting segment exits.
- **Hybrid segment subgraphs** (`_is_segment_subgraph`) must never be passed to `prune_ir_graph`.

Torch mask exemption uses `compute_perceptron_io_exemption_indices` (same rules, plus `is_encoding_layer` for host-side encoders).

## Dependencies

- **Internal**: `mapping.ir`, `models.ttfs_kernels` (liveness)
- **External**: `numpy`

## Dependents

- `mapping.packing.hybrid_build`, verification steps, layout verifier, `transformations.pruning`, `tuning.tuners.pruning`.

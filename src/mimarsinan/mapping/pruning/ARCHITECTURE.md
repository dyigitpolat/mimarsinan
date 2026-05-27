# mapping/pruning/ — IR Pruning and Liveness

Graph pruning, liveness analysis, and segmentation metadata for hybrid mapping.

## Key modules

| File | Symbols | Role |
|------|---------|------|
| `ir_pruning.py` | `prune_ir_graph` | Main prune pass |
| `ir_pruning_analysis.py` | `compute_graph_io_exemption` | IO exemption analysis |
| `ir_liveness.py` | Liveness helpers | Row/col liveness for pruning |
| `liveness_semantics.py` | TTFS/liveness rules | Activation semantics for prune decisions |
| `pruning_propagation.py` | `compute_propagated_pruned_rows_cols` | Propagate prune masks |
| `pruning_graph_propagation.py` | `compute_global_pruned_sets` | Global prune sets |
| `pruning_apply.py` | Apply helpers | Apply prune results to structures |
| `ir_segmentation.py` | `build_ir_consumed_by` | Segment boundary metadata |

## Dependencies

- **Internal**: `mapping.ir`, `models.ttfs_kernels` (liveness)
- **External**: `numpy`

## Dependents

- `mapping.packing.hybrid_build`, verification steps, layout verifier.

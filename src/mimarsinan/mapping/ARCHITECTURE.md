# mapping/ -- Model-to-Hardware Mapping

Converts PyTorch models to an intermediate representation (IR) and then packs
the IR into physical hardware cores.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `ir.py` | `IRSource`, `IRNode`, `NeuralCore`, `ComputeOp`, `IRGraph`, `WeightBank` | Unified IR: directed graph of neural cores and compute operations. `WeightBank` stores shared weights for conv-style layers; `NeuralCore` can either own its `core_matrix` or reference a bank via `weight_bank_id`. Pruning provenance: `NeuralCore.perceptron_index`, `perceptron_output_slice`, `perceptron_input_slice`; `WeightBank.perceptron_index` — used by `get_initial_pruning_masks_from_model` to apply model pruning masks to tiled FC and bank-backed cores. |
| `ir_mapping.py` | `IRMapping` | Converts `ModelRepresentation` (mapper graph) to `IRGraph`; handles output/axon tiling. Provides `register_weight_bank()` and `add_shared_neural_core()` for conv-style shared-weight mapping. Accepts optional `perceptron_index` (and `perceptron_output_slice` / `perceptron_input_slice` for tiled FC) so pruning can apply model masks to tiled and bank-backed IR. |
| `mapping_utils.py` | `Mapper`, `ModelRepresentation`, `SoftCoreMapping`, `PerceptronMapper`, `Conv2DPerceptronMapper`, ... | Mapper hierarchy: dual-purpose DAG for forward pass and hardware mapping. `ModelRepresentation.assign_perceptron_indices()` sets `perceptron_index` on each mapper that owns perceptrons (same order as `get_perceptrons()`). `PerceptronMapper` and `Conv2DPerceptronMapper` pass `perceptron_index` into `map_fc` / `register_weight_bank` when mapping to IR. |
| `softcore_mapping.py` | `SoftCore`, `HardCore`, `HardCoreMapping`, `compact_soft_core_mapping` | Logical-to-physical core representations and packing. `compact_soft_core_mapping` uses `pruned_row_mask` / `pruned_col_mask` to drop rows/cols and reindex; raises if compaction would remove every output ref. Two-way traceability: `neuron_mapping` (soft→hard); `soft_core_placements_per_hard_core` (hard→soft) for overlay/UI. |
| `core_packing.py` | `greedy_pack_softcores` | Generic best-fit greedy bin-packing algorithm |
| `hybrid_hardcore_mapping.py` | `HybridHardCoreMapping`, `HybridStage`, `SegmentIOSlice`, `build_hybrid_hard_core_mapping` | Multi-segment deployable program with state-buffer I/O |
| `chip_latency.py` | `ChipLatency` | Calculates chip simulation latency from core graph; raises if mapping has no output_sources. |
| `ir_latency.py` | `IRLatency` | Computes per-node latency tiers in the IR graph |
| `ir_pruning.py` | `prune_ir_graph`, `get_initial_pruning_masks_from_model`, `get_neural_segments`, `compute_segment_io_exemption` | Eliminates zeroed/pruned rows and columns from the IR; uses model pruning maps when available (else zero-threshold); When model masks are provided, propagation always runs (exemption-aware). When no initial masks are given, propagative pruning is always used (delegated to `pruning_propagation`). When neural core count differs from perceptron count (tiled IR), model masks are used only when cores/banks have `perceptron_index` set (sliced by `perceptron_output_slice`/`perceptron_input_slice`); otherwise returns empty masks so only zero-threshold pruning is used. Output nodes use only zero-threshold. For bank-backed cores, Phase 5 sets per-node `pre_pruning_heatmap` and `pruned_row_mask` / `pruned_col_mask` sliced to the node’s effective matrix (by `weight_row_slice` when present) so soft-core compaction and GUI pre/post views see consistent shapes. Segment I/O exemption: per segment, rows fed by segment input (node_id=-2 or external) and columns feeding segment output are never pruned. |
| `pruning_propagation.py` | `compute_propagated_pruned_rows_cols` | Single place for propagative pruning fixpoint (row only feeds pruned cols, col only receives from pruned rows). Exempt indices (`exempt_rows`/`exempt_cols`) are never added at init or in the fixpoint. Used by `ir_pruning` for owned-weight cores and weight banks. |
| `per_source_scales.py` | `compute_per_source_scales` | Traverses mapper graph to set per-input-channel `per_input_scales` on each perceptron; handles branching (concat) and dimension-rearranging mappers (falls back to mean when channel counts don't align) |
| `ir_source_spans.py` | `IRSourceSpan`, `compress_ir_sources` | Range-compressed IR source representations for efficient simulation |
| `spike_source_spans.py` | `SpikeSourceSpan`, `compress_spike_sources` | Range-compressed spike source representations |

### Subdirectory

| Directory | Purpose |
|-----------|---------|
| `layout/` | Shape-only layout estimation for architecture search (no weights) |

## Dependencies

- **Internal**: `code_generation.cpp_chip_model` (`SpikeSource`), `models.layers`, `models.perceptron_mixer.perceptron` (`Perceptron`), `transformations` (`PerceptronTransformer`, `TensorQuantization`).
- **External**: `torch`, `numpy`, `einops`.

## Dependents

- `models` imports `Mapper` classes for building mapper graphs, IR types for spiking simulation
- `pipelining` imports `IRMapping`, `IRGraph`, `NeuralCore` for mapping steps
- `chip_simulation` imports mappings for nevresim code generation
- `visualization` imports IR and mapping types for DOT generation
- `tuning` imports `NeuralCore` for CoreFlow tuning
- `search` imports layout utilities for architecture search
- `gui` imports IR types for snapshot extraction

## Exported API (\_\_init\_\_.py)

Core IR types (including `WeightBank`), mapping classes, packing utilities, `compute_per_source_scales`, `prune_ir_graph`, and `compute_propagated_pruned_rows_cols`.
`mapping_utils` (the large mapper hierarchy) is intentionally **not** re-exported at
the package level due to its size and star-import patterns; import directly from
`mapping.mapping_utils`.

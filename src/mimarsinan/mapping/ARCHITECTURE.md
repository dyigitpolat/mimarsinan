# mapping/ -- Model-to-Hardware Mapping

Converts PyTorch models to an intermediate representation (IR) and then packs
the IR into physical hardware cores.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `ir.py` | `IRSource`, `IRNode`, `NeuralCore`, `ComputeOp`, `IRGraph`, `WeightBank` | Unified IR: directed graph of neural cores and compute operations. `WeightBank` stores shared weights for conv-style layers; `NeuralCore` can either own its `core_matrix` or reference a bank via `weight_bank_id`. Pruning provenance: `NeuralCore.perceptron_index`, `perceptron_output_slice`, `perceptron_input_slice`; `WeightBank.perceptron_index` — used by `get_initial_pruning_masks_from_model` to apply model pruning masks to tiled FC and bank-backed cores. |
| `ir_mapping.py` | `IRMapping` | Converts `ModelRepresentation` (mapper graph) to `IRGraph`; handles output/axon tiling. Provides `register_weight_bank()` and `add_shared_neural_core()` for conv-style shared-weight mapping. Tiling is always enabled for wide layers; `allow_core_coalescing` selects between psum decomposition (2N+1 cores) and wide-core coalescing (N+1 cores). **Bias modes**: `hardware_bias=True` stores bias in `NeuralCore.hardware_bias` (dedicated register, no always-on axon row); `hardware_bias=False` uses legacy always-on row mode. **Note**: IR thresholds are always unit (1.0) as `MapperRepr` provides weights already normalized by activation and input scales; `parameter_scale` is preserved as metadata for hardware scaling. |
| `mapping_utils.py` | `Mapper`, `PerceptronMapper`, `Conv2DPerceptronMapper`, ... (re-exports from `soft_core_mapper`, `model_representation`, `chip_export`) | Mapper hierarchy: dual-purpose DAG for forward pass and hardware mapping. Re-exports `ModelRepresentation` (from `model_representation`), `SoftCoreMapping`/`map_mm` (from `soft_core_mapper`), `hard_cores_to_chip`/`generate_core_weights`/`to_numpy` (from `chip_export`). `ModelRepresentation.assign_perceptron_indices()` sets `perceptron_index` on each mapper that owns perceptrons. |
| `soft_core_mapper.py` | `SoftCoreMapping`, `map_mm` | Maps `ModelRepresentation` to list of SoftCores; lazy imports for `compress_spike_sources`. Consumed by `ir` (lazy) and re-exported by `mapping_utils`. |
| `model_representation.py` | `ModelRepresentation` | DAG wrapper: exec order, perceptron groups, `assign_perceptron_indices`, `map_to_ir`. Re-exported by `mapping_utils`. |
| `chip_export.py` | `hard_cores_to_chip`, `generate_core_weights`, `generate_core_connection_info`, `to_numpy` | Converts `HardCoreMapping` → `ChipModel` for nevresim. Single consumer: `chip_simulation.nevresim_driver`. |
| `softcore_mapping.py` | `SoftCore`, `HardCore`, `HardCoreMapping`, `compact_soft_core_mapping` | Logical-to-physical core representations and packing. **Bias chain**: `SoftCore.hardware_bias` and `HardCore.hardware_bias` carry dedicated per-neuron bias arrays through the packing pipeline; `HardCore.add_softcore()` copies bias into the correct neuron slice; `compact_soft_core_mapping` compacts `hardware_bias` alongside pruned columns. When `hardware_bias` is set, no always-on axon row exists. Legacy mode: `HardCore.has_bias_capability` (from `cores_config["has_bias"]`); codegen folds the last matrix row into per-neuron bias. Two-way traceability: `neuron_mapping` (soft→hard); `soft_core_placements_per_hard_core` (hard→soft) for overlay/UI. **Fused cores**: when the packer fuses multiple physical HardCores via `fuse_hardcores`, the returned fused HardCore has optional `fused_component_axons` (list of axon counts per component) set in `HardCoreMapping.map()` for GUI boundary and badge display. |
| `core_packing.py` | `greedy_pack_softcores`, `pre_allocate_coalescing_groups` | Generic best-fit greedy bin-packing algorithm; `pre_allocate_coalescing_groups` reserves dedicated `HardCore`s for coalescing partial softcores (which occupy the full axon width and cannot share a core via diagonal packing) before the main greedy pass |
| `hybrid_hardcore_mapping.py` | `HybridHardCoreMapping`, `HybridStage`, `SegmentIOSlice`, `build_hybrid_hard_core_mapping` | Multi-segment deployable program with state-buffer I/O |
| `chip_latency.py` | `ChipLatency` | Calculates chip simulation latency from core graph; raises if mapping has no output_sources. |
| `ir_latency.py` | `IRLatency` | Computes per-node latency tiers in the IR graph |
| `ir_pruning_analysis.py` | `get_neural_segments`, `compute_segment_io_exemption` | Pure graph queries: neural segments and per-node I/O exemption. Used by `ir_pruning`. |
| `ir_pruning.py` | `prune_ir_graph`, `get_initial_pruning_masks_from_model` (segment helpers in `ir_pruning_analysis`) | Eliminates zeroed/pruned rows and columns from the IR; uses model pruning maps when available (else zero-threshold); When model masks are provided, propagation always runs (exemption-aware). When no initial masks are given, propagative pruning is always used (delegated to `pruning_propagation`). When neural core count differs from perceptron count (tiled IR), model masks are used only when cores/banks have `perceptron_index` set (sliced by `perceptron_output_slice`/`perceptron_input_slice`); otherwise returns empty masks so only zero-threshold pruning is used. Output nodes use only zero-threshold. For bank-backed cores, Phase 5 sets per-node `pre_pruning_heatmap` and `pruned_row_mask` / `pruned_col_mask` sliced to the node’s effective matrix (by `weight_row_slice` when present) so soft-core compaction and GUI pre/post views see consistent shapes. Segment I/O exemption: only graph-input rows (node_id=-2) and columns that feed graph output (output_sources) are exempt. Rows fed by the previous segment or ComputeOp are not exempt (so axon/row pruning compacts); columns consumed by the next segment or ComputeOp are not exempt (so neuron/column pruning compacts). |
| `pruning_propagation.py` | `compute_propagated_pruned_rows_cols` | Single place for propagative pruning fixpoint (row only feeds pruned cols, col only receives from pruned rows). Exempt indices (`exempt_rows`/`exempt_cols`) are never added at init or in the fixpoint. Used by `ir_pruning` for owned-weight cores and weight banks. |
| `per_source_scales.py` | `compute_per_source_scales` | Traverses mapper graph to set per-input-channel `per_input_scales` on each perceptron; handles branching (concat) and dimension-rearranging mappers (falls back to mean when channel counts don't align) |
| `ir_source_spans.py` | `IRSourceSpan`, `compress_ir_sources` | Range-compressed IR source representations for efficient simulation |
| `spike_source_spans.py` | `SpikeSourceSpan`, `compress_spike_sources` | Range-compressed spike source representations |
| `mapping_verifier.py` | `verify_soft_core_mapping`, `verify_hardware_config`, `MappingVerificationResult` | Layout mapping verification; hardware config check. With multiple core types, at least one type must fit the largest softcore. `MappingVerificationResult` also carries `host_side_segment_count` (compute-only runs between / around neural segments) and `layout_preview` (compact input/host/latency-group/output flow summary for the wizard). `verify_hardware_config` returns a `"stats"` dict (from `LayoutVerificationStats.to_dict()`) alongside feasibility. |
| `layout_verification_stats.py` | `LayoutVerificationStats`, `build_layout_verification_stats`, `build_stats_from_packing_result` | Pure stats module: computes total and per-core wasted-axon/neuron percentages, mapped-parameter utilization, coalescing/splitting counts, and layout-derived neural-segment summaries from a `LayoutPackingResult`. Segment latency summary reports latency groups per neural segment, not absolute global depth indices. UI-agnostic; reusable by wizard, monitor, search, or reports. |
| `hw_config_suggester.py` | `suggest_hardware_config`, `suggest_hardware_config_for_model`, `HardwareSuggestion` | Two-type hardware suggester: H×W and W×H (or H×H and W×H when coalescing); smallest H,W such that >50% of used cores host ≥4 softcores. |

### Perceptron packaging rule

Perceptron packaging follows the pattern **MM+ → BN? → ACT**. Two predicates in
`mappers/base.py` derive from `CHIP_SUPPORTED_ACTIVATIONS`:

- `is_chip_targeted_activation` — True for all except Identity; controls
  `owned_perceptron_groups()` (pipeline processing, scale propagation).
- `is_chip_supported_activation` — True only for ReLU-like; controls actual IR
  mapping (NeuralCore vs host-side ComputeOp).

There is no separate host-side activation list; host-side is derived as
"not chip-targeted" (Identity only).

**Mapper eligibility contract**: Every concrete mapper type
(`PerceptronMapper`, `Conv2DPerceptronMapper`, `Conv1DPerceptronMapper`) implements
`owned_perceptron_groups()` using `is_chip_targeted_activation()`. This means
`ModelRepresentation.get_perceptrons()` always returns only chip-targeted
perceptrons. Downstream pipeline steps and tuners can rely on this contract and
do not need to special-case `Identity` activation.

**Layout-pass contract**: `LayoutIRMapping` sets `_is_layout_pass = True`.
Mappers check this flag in `_map_to_ir` to route chip-targeted perceptrons
(e.g. GELU) to `map_fc` (NeuralCore estimate) rather than `add_linear_compute_op`.
This gives correct hardware sizing because adaptable activations (GELU → ReLU)
will be chip-supported by the time the actual IR mapping runs. The actual
`IRMapping` does not set `_is_layout_pass`, so GELU still becomes a ComputeOp
there.

### Subdirectory

| Directory | Purpose |
|-----------|---------|
| `mappers/` | Mapper hierarchy split by role: `base`, `structural`, `perceptron`, `leading_dim`, `pooling`, `conv`, `transformer`. Re-exported by `mapping_utils.py` for backward compatibility. |
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

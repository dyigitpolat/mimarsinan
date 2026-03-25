# mapping/ -- Model-to-Hardware Mapping

Converts PyTorch models to an intermediate representation (IR) and then packs
the IR into physical hardware cores.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `ir.py` | `IRSource`, `IRNode`, `NeuralCore`, `ComputeOp`, `IRGraph`, `WeightBank` | Unified IR: directed graph of neural cores and compute operations. `WeightBank` stores shared weights for conv-style layers; `NeuralCore` can either own its `core_matrix` or reference a bank via `weight_bank_id`. Pruning provenance: `NeuralCore.perceptron_index`, `perceptron_output_slice`, `perceptron_input_slice`; `WeightBank.perceptron_index` — used by `get_initial_pruning_masks_from_model` to apply model pruning masks to tiled FC and bank-backed cores. |
| `ir_mapping.py` | `IRMapping` | Converts `ModelRepresentation` (mapper graph) to `IRGraph`; handles output/axon tiling. Provides `register_weight_bank()` and `add_shared_neural_core()` for conv-style shared-weight mapping. Tiling is always enabled for wide layers; `allow_core_coalescing` selects between psum decomposition (2N+1 cores) and wide-core coalescing (N+1 cores). **Bias modes**: `hardware_bias=True` stores bias in `NeuralCore.hardware_bias` (dedicated register, no always-on axon row); `hardware_bias=False` uses legacy always-on row mode. **Note**: IR thresholds are always unit (1.0) as `MapperRepr` provides weights already normalized by activation and input scales; `parameter_scale` is preserved as metadata for hardware scaling. Structural decisions (bias counting, wide-layer detection, psum params) delegated to `mapping_structure`. |
| `mapping_structure.py` | `compute_core_input_count`, `compute_fc_tiling_mode`, `compute_psum_params`, `PsumParams` | Shared structural decision helpers used by both `LayoutIRMapping` and `IRMapping`. Ensures identical bias-axon counting, wide-layer detection thresholds, and psum decomposition parameters across both backends. Pure functions, no mapping state. |
| `mapping_utils.py` | `Mapper`, `PerceptronMapper`, `Conv2DPerceptronMapper`, ... (re-exports from `soft_core_mapper`, `model_representation`, `chip_export`) | Mapper hierarchy: dual-purpose DAG for forward pass and hardware mapping. Re-exports `ModelRepresentation` (from `model_representation`), `SoftCoreMapping`/`map_mm` (from `soft_core_mapper`), `hard_cores_to_chip`/`generate_core_weights`/`to_numpy` (from `chip_export`). `ModelRepresentation.assign_perceptron_indices()` sets `perceptron_index` on each mapper that owns perceptrons. |
| `soft_core_mapper.py` | `SoftCoreMapping`, `map_mm` | Maps `ModelRepresentation` to list of SoftCores; lazy imports for `compress_spike_sources`. Consumed by `ir` (lazy) and re-exported by `mapping_utils`. |
| `model_representation.py` | `ModelRepresentation` | DAG wrapper: exec order, perceptron groups, `assign_perceptron_indices`, `map_to_ir`. Re-exported by `mapping_utils`. |
| `chip_export.py` | `hard_cores_to_chip`, `generate_core_weights`, `generate_core_connection_info`, `to_numpy` | Converts `HardCoreMapping` → `ChipModel` for nevresim. Single consumer: `chip_simulation.nevresim_driver`. |
| `softcore_mapping.py` | `SoftCore`, `HardCore`, `HardCoreMapping`, `compact_soft_core_mapping` | Logical-to-physical core representations and packing. **Bias chain**: `SoftCore.hardware_bias` and `HardCore.hardware_bias` carry dedicated per-neuron bias arrays through the packing pipeline; `HardCore.add_softcore()` copies bias into the correct neuron slice; `compact_soft_core_mapping` compacts `hardware_bias` alongside pruned columns. When `hardware_bias` is set, no always-on axon row exists. Legacy mode: `HardCore.has_bias_capability` (from `cores_config["has_bias"]`); codegen folds the last matrix row into per-neuron bias. Two-way traceability: `neuron_mapping` (soft→hard); `soft_core_placements_per_hard_core` (hard→soft) for overlay/UI. **Fused cores**: when the packer fuses multiple physical HardCores via `fuse_hardcores`, the returned fused HardCore has optional `fused_component_axons` (list of axon counts per component) set in `HardCoreMapping.map()` for GUI boundary and badge display. |
| `core_packing.py` | `greedy_pack_softcores`, `pre_allocate_coalescing_groups` | Generic best-fit greedy bin-packing algorithm; `pre_allocate_coalescing_groups` reserves dedicated `HardCore`s for coalescing partial softcores (which occupy the full axon width and cannot share a core via diagonal packing) before the main greedy pass |
| `hybrid_hardcore_mapping.py` | `HybridHardCoreMapping`, `HybridStage`, `SegmentIOSlice`, `build_hybrid_hard_core_mapping` | Multi-segment deployable program with state-buffer I/O. `HybridStage` carries optional `schedule_segment_index` / `schedule_pass_index` metadata for scheduled mapping. When `allow_scheduling=True`, `build_hybrid_hard_core_mapping` allocates a **fresh** hardware core pool per pass (same physical cores reprogrammed) instead of a single shared pool, enabling models that need more cores than available. `node_activation_scales` maps IR node_id → activation_scale float; used by `SpikingHybridCoreFlow` in TTFS mode to rescale ComputeOp inputs/outputs so bias terms remain correct. |
| `schedule_partitioner.py` | `partition_segment_into_passes`, `estimate_passes_for_layout`, `effective_core_budget` | Partitions a neural segment's cores into sequential schedule passes via a **unified** generic algorithm (`_partition_with_latencies`) that accepts any object with `get_input_count()`/`get_output_count()`. Both `partition_segment_into_passes` (NeuralCore) and `estimate_passes_for_layout` (LayoutSoftCoreSpec) delegate to the same algorithm, ensuring identical pass counts for identical shapes. `effective_core_budget` computes the 0.8× heterogeneous-discount budget used by both the wizard verifier and hybrid mapping builder. `_compute_core_latencies` saves/restores original `NeuralCore.latency` to avoid mutation. |
| `chip_latency.py` | `ChipLatency` | Calculates chip simulation latency from core graph; raises if mapping has no output_sources. |
| `ir_latency.py` | `IRLatency` | Computes per-node latency tiers in the IR graph |
| `ir_pruning_analysis.py` | `get_neural_segments`, `compute_segment_io_exemption` | Pure graph queries: neural segments and per-node I/O exemption. Used by `ir_pruning`. |
| `ir_pruning.py` | `prune_ir_graph`, `get_initial_pruning_masks_from_model` (segment helpers in `ir_pruning_analysis`) | Eliminates zeroed/pruned rows and columns from the IR; uses model pruning maps when available (else zero-threshold); When model masks are provided, propagation always runs (exemption-aware). When no initial masks are given, propagative pruning is always used (delegated to `pruning_propagation`). When neural core count differs from perceptron count (tiled IR), model masks are used only when cores/banks have `perceptron_index` set (sliced by `perceptron_output_slice`/`perceptron_input_slice`); otherwise returns empty masks so only zero-threshold pruning is used. Output nodes use only zero-threshold. For bank-backed cores, Phase 5 sets per-node `pre_pruning_heatmap` and `pruned_row_mask` / `pruned_col_mask` sliced to the node’s effective matrix (by `weight_row_slice` when present) so soft-core compaction and GUI pre/post views see consistent shapes. Segment I/O exemption: only graph-input rows (node_id=-2) and columns that feed graph output (output_sources) are exempt. Rows fed by the previous segment or ComputeOp are not exempt (so axon/row pruning compacts); columns consumed by the next segment or ComputeOp are not exempt (so neuron/column pruning compacts). |
| `pruning_propagation.py` | `compute_propagated_pruned_rows_cols` | Single place for propagative pruning fixpoint (row only feeds pruned cols, col only receives from pruned rows). Exempt indices (`exempt_rows`/`exempt_cols`) are never added at init or in the fixpoint. Used by `ir_pruning` for owned-weight cores and weight banks. |
| `per_source_scales.py` | `compute_per_source_scales` | Traverses mapper graph to set per-input-channel `per_input_scales` on each perceptron; handles branching (concat) and dimension-rearranging mappers (falls back to mean when channel counts don't align) |
| `ir_source_spans.py` | `IRSourceSpan`, `compress_ir_sources` | Range-compressed IR source representations for efficient simulation |
| `spike_source_spans.py` | `SpikeSourceSpan`, `compress_spike_sources` | Range-compressed spike source representations |
| `mapping_verifier.py` | `verify_soft_core_mapping`, `verify_hardware_config`, `MappingVerificationResult` | Layout mapping verification; hardware config check. `verify_soft_core_mapping` accepts `allow_core_coalescing` and `hardware_bias` flags that are forwarded to `LayoutIRMapping` so layout-level core counts match actual IR mapping for wide FC layers. With multiple core types, at least one type must fit the largest softcore. `MappingVerificationResult` also carries `host_side_segment_count` (compute-only runs between / around neural segments) and `layout_preview` (compact input/host/latency-group/output flow summary for the wizard). `verify_hardware_config` returns a `"stats"` dict (from `LayoutVerificationStats.to_dict()`) alongside feasibility. |
| `layout_verification_stats.py` | `LayoutVerificationStats`, `build_layout_verification_stats`, `build_stats_from_packing_result` | Pure stats module: computes total and per-core wasted-axon/neuron percentages, mapped-parameter utilization, coalescing/splitting counts, and layout-derived neural-segment summaries from a `LayoutPackingResult`. Segment latency summary reports latency groups per neural segment, not absolute global depth indices. UI-agnostic; reusable by wizard, monitor, search, or reports. |
| `hw_config_suggester.py` | `suggest_hardware_config`, `suggest_hardware_config_scheduled`, `suggest_hardware_config_for_model`, `HardwareSuggestion` | Two-type hardware suggester: H×W and W×H (or H×H and W×H when coalescing); smallest H,W such that >50% of used cores host ≥4 softcores. `suggest_hardware_config_scheduled` explores the core-count ↔ pass-count tradeoff when scheduling is enabled, using cost model `area × passes^latency_weight`. `HardwareSuggestion` includes `num_passes` and `estimated_latency_multiplier`. |

### Perceptron packaging rule

Perceptron packaging follows the pattern **MM+ → BN? → ACT**. A single predicate
in `mappers/base.py` controls the packaging decision:

- `is_perceptron_activation(perceptron)` — True if the perceptron has a real
  (non-Identity) activation. Uses `isinstance` check, not a hardcoded activation set.

Any detected nonlinearity (ReLU, GELU, LeakyReLU, etc.) qualifies as a perceptron
and maps to a NeuralCore. The adaptation pipeline converts all activations to
LeakyGradReLU before deployment. Identity (no activation detected) produces a
host-side linear ComputeOp.

**Mapper eligibility contract**: Every concrete mapper type
(`PerceptronMapper`, `Conv2DPerceptronMapper`, `Conv1DPerceptronMapper`) implements
`owned_perceptron_groups()` using `is_perceptron_activation()`. This means
`ModelRepresentation.get_perceptrons()` always returns only perceptrons with
nonlinear activations. Downstream pipeline steps and tuners can rely on this
contract and do not need to special-case `Identity` activation.

### Scheduled mapping

When `allow_scheduling=True` in the merged pipeline config (typically from `deployment_parameters`), `build_hybrid_hard_core_mapping`
splits neural segments into multiple **schedule passes** that execute sequentially,
reusing the same physical hardware cores.  This trades latency for chip area.

**Key design**: schedule passes are additional `HybridStage` entries — the existing
state-buffer execution model in `SpikingHybridCoreFlow` handles inter-pass data
handoff identically to inter-segment handoff, requiring zero changes to the simulator.

**Partitioning algorithm** (`schedule_partitioner.py`):
1. Compute latency per core within the segment (save/restore to avoid mutating originals).
2. Group by latency; greedily assign groups to passes (increasing latency order).
3. If a single latency group exceeds available cores, bin-pack its atomic units
   using first-fit-decreasing.  Coalescing groups may be split across passes —
   the state buffer handles inter-pass data flow for partial-sum fragments.
4. `effective_core_budget()` computes the budget: for heterogeneous core configs
   the total is reduced by 20% to account for per-type scarcity.  Both the wizard
   verifier and the hybrid mapping builder call this function so they agree.
5. The layout-level estimator (`estimate_passes_for_layout`) and the build-time
   partitioner (`partition_segment_into_passes`) share the same core algorithm
   (`_partition_with_latencies`) — identical shapes and budget produce identical
   pass counts.

**HW config suggester**: `suggest_hardware_config_scheduled` searches pass counts
1..`max_schedule_passes`, finds minimum cores per pass count, and picks the
configuration minimizing `core_area × passes^latency_weight`.

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

Core IR types (including `WeightBank`), mapping classes, packing utilities, shared structural helpers (`compute_core_input_count`, `compute_fc_tiling_mode`, `compute_psum_params`), `compute_per_source_scales`, `prune_ir_graph`, and `compute_propagated_pruned_rows_cols`.
`mapping_utils` (the large mapper hierarchy) is intentionally **not** re-exported at
the package level due to its size and star-import patterns; import directly from
`mapping.mapping_utils`.

# mapping/ -- Model-to-Hardware Mapping

Converts PyTorch models to an intermediate representation (IR) and then packs
the IR into physical hardware cores.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `ir.py` | `IRSource`, `IRNode`, `NeuralCore`, `ComputeOp`, `IRGraph`, `WeightBank` | Unified IR: directed graph of neural cores and compute operations. `WeightBank` stores shared weights for conv-style layers; `NeuralCore` can either own its `core_matrix` or reference a bank via `weight_bank_id`. Pruning provenance: `NeuralCore.perceptron_index`, `perceptron_output_slice`, `perceptron_input_slice`; `WeightBank.perceptron_index` — used by `get_initial_pruning_masks_from_model` to apply model pruning masks to tiled FC and bank-backed cores. |
| `coalescing.py` | `CANONICAL_KEY`, `coalescing_config_errors`, `resolve_allow_coalescing`, `normalize_coalescing_config`, `CoalescingConfigError` | Single flag `allow_coalescing` for layout, packing, and scheduling; deprecated names (`allow_core_coalescing`, `allow_axon_coalescing`, `allow_axon_tiling`) are rejected. |
| `ir_mapping.py` | `IRMapping` | Converts `ModelRepresentation` (mapper graph) to `IRGraph`; handles output/axon tiling. Provides `register_weight_bank()` and `add_shared_neural_core()` for conv-style shared-weight mapping. Tiling is always enabled for wide layers; `allow_coalescing` selects between psum decomposition (2N+1 cores) and wide-core coalescing (N+1 cores). **Bias modes**: `hardware_bias=True` stores bias in `NeuralCore.hardware_bias` (dedicated register, no always-on axon row); `hardware_bias=False` uses legacy always-on row mode. **Note**: IR thresholds are always unit (1.0) as `MapperRepr` provides weights already normalized by activation and input scales; `parameter_scale` is preserved as metadata for hardware scaling. Structural decisions (bias counting, wide-layer detection, psum params) delegated to `mapping_structure`. |
| `mapping_structure.py` | `compute_core_input_count`, `compute_fc_tiling_mode`, `compute_psum_params`, `PsumParams` | Shared structural decision helpers used by both `LayoutIRMapping` and `IRMapping`. Ensures identical bias-axon counting, wide-layer detection thresholds, and psum decomposition parameters across both backends. Pure functions, no mapping state. |
| `mapping_utils.py` | `Mapper`, `PerceptronMapper`, `Conv2DPerceptronMapper`, ... (re-exports from `soft_core_mapper`, `model_representation`, `chip_export`) | Mapper hierarchy: dual-purpose DAG for forward pass and hardware mapping. Re-exports `ModelRepresentation` (from `model_representation`), `SoftCoreMapping`/`map_mm` (from `soft_core_mapper`), `hard_cores_to_chip`/`generate_core_weights`/`to_numpy` (from `chip_export`). `ModelRepresentation.assign_perceptron_indices()` sets `perceptron_index` on each mapper that owns perceptrons. |
| `soft_core_mapper.py` | `SoftCoreMapping`, `map_mm` | Maps `ModelRepresentation` to list of SoftCores; lazy imports for `compress_spike_sources`. Consumed by `ir` (lazy) and re-exported by `mapping_utils`. |
| `model_representation.py` | `ModelRepresentation` | DAG wrapper: exec order, perceptron groups, `assign_perceptron_indices`, `map_to_ir`. Re-exported by `mapping_utils`. |
| `chip_export.py` | `hard_cores_to_chip`, `generate_core_weights`, `generate_core_connection_info`, `to_numpy` | Converts `HardCoreMapping` → `ChipModel` for nevresim. Single consumer: `chip_simulation.nevresim_driver`. |
| `softcore_mapping.py` | `SoftCore`, `HardCore`, `HardCoreMapping`, `compact_soft_core_mapping` | Logical-to-physical core representations and packing. **Bias chain**: `SoftCore.hardware_bias` and `HardCore.hardware_bias` carry dedicated per-neuron bias arrays through the packing pipeline; `HardCore.add_softcore()` copies bias into the correct neuron slice; `compact_soft_core_mapping` compacts `hardware_bias` alongside pruned columns. When `hardware_bias` is set, no always-on axon row exists. Legacy mode: `HardCore.has_bias_capability` (from `cores_config["has_bias"]`); codegen folds the last matrix row into per-neuron bias. Two-way traceability: `neuron_mapping` (soft→hard); `soft_core_placements_per_hard_core` (hard→soft) for overlay/UI. **Fused cores**: when the packer fuses multiple physical HardCores via `fuse_hardcores`, the returned fused HardCore has optional `fused_component_axons` (list of axon counts per component) set in `HardCoreMapping.map()` for GUI boundary and badge display. |
| `core_packing.py` | `greedy_pack_softcores`, `pre_allocate_coalescing_groups` | Generic best-fit greedy bin-packing algorithm; `pre_allocate_coalescing_groups` reserves dedicated `HardCore`s for coalescing partial softcores (which occupy the full axon width and cannot share a core via diagonal packing) before the main greedy pass |
| `hybrid_hardcore_mapping.py` | `HybridHardCoreMapping`, `HybridStage`, `SegmentIOSlice`, `build_hybrid_hard_core_mapping`, `_validate_coalescing_budget` | Multi-segment deployable program with state-buffer I/O. **Each neural segment → exactly one `HybridStage` of `kind="neural"`**. Segment boundaries come solely from ComputeOp sync barriers (inserted upstream by the layout mapper); the hard-core mapper does not sub-split within a segment. When `allow_scheduling=True`, each barrier-separated segment is packed onto a fresh hardware pool (same physical cores reprogrammed between segments); within a segment the combined cores must fit in one pass — if the packer runs out of cores, the `RuntimeError` is propagated unchanged so the infeasibility is loud. `_flush_neural_segment_scheduled` performs this single-pass flush; `_validate_coalescing_budget` enforces that every wide NeuralCore's coalescing group fits within a single core type's count. `node_activation_scales` maps IR node_id → activation_scale float; used by `SpikingHybridCoreFlow` in TTFS mode to rescale ComputeOp inputs/outputs so bias terms remain correct. |
| `schedule_partitioner.py` | `partition_segment_into_passes`, `estimate_passes_for_layout`, `estimate_passes_for_layout_validated`, `effective_core_budget` | Segment-level pass accounting. Historical multi-pass sub-splitting has been removed: every neural segment becomes exactly one pass, and segment boundaries are set by the layout mapper alone. `partition_segment_into_passes` is the identity on the segment's cores (preserved for ABI). `estimate_passes_for_layout` reports one pass per distinct `segment_id`; `estimate_passes_for_layout_validated` additionally runs `pack_layout` on each segment to report feasibility. `effective_core_budget` retains the 0.8× heterogeneous-discount used by the wizard and hard-core mapper. |
| `chip_latency.py` | `ChipLatency` | Calculates chip simulation latency from core graph; raises if mapping has no output_sources. |
| `ir_latency.py` | `IRLatency` | Computes per-node latency tiers in the IR graph |
| `ir_pruning_analysis.py` | `get_neural_segments`, `compute_segment_io_exemption` | Pure graph queries: neural segments and per-node I/O exemption. Used by `ir_pruning`. |
| `ir_pruning.py` | `prune_ir_graph`, `get_initial_pruning_masks_from_model` (segment helpers in `ir_pruning_analysis`) | Eliminates zeroed/pruned rows and columns from the IR; uses model pruning maps when available (else zero-threshold); When model masks are provided, propagation always runs (exemption-aware). When no initial masks are given, propagative pruning is always used (delegated to `pruning_propagation`). When neural core count differs from perceptron count (tiled IR), model masks are used only when cores/banks have `perceptron_index` set (sliced by `perceptron_output_slice`/`perceptron_input_slice`); otherwise returns empty masks so only zero-threshold pruning is used. Output nodes use only zero-threshold. For bank-backed cores, Phase 5 sets per-node `pre_pruning_heatmap` and `pruned_row_mask` / `pruned_col_mask` sliced to the node’s effective matrix (by `weight_row_slice` when present) so soft-core compaction and GUI pre/post views see consistent shapes. Segment I/O exemption: only graph-input rows (node_id=-2) and columns that feed graph output (output_sources) are exempt. Rows fed by the previous segment or ComputeOp are not exempt (so axon/row pruning compacts); columns consumed by the next segment or ComputeOp are not exempt (so neuron/column pruning compacts). |
| `pruning_propagation.py` | `compute_propagated_pruned_rows_cols` | Single place for propagative pruning fixpoint (row only feeds pruned cols, col only receives from pruned rows). Exempt indices (`exempt_rows`/`exempt_cols`) are never added at init or in the fixpoint. Used by `ir_pruning` for owned-weight cores and weight banks. |
| `per_source_scales.py` | `compute_per_source_scales`, `_broadcast_scale_pair` | Traverses mapper graph to set per-input-channel `per_input_scales` on each perceptron; handles branching (concat, add) and dimension-rearranging mappers (falls back to mean when channel counts don't align). `_broadcast_scale_pair` reconciles scale vectors of different lengths for `AddMapper` using `repeat_interleave` (when divisible) or mean-fill fallback |
| `ir_source_spans.py` | `IRSourceSpan`, `compress_ir_sources` | Range-compressed IR source representations for efficient simulation |
| `spike_source_spans.py` | `SpikeSourceSpan`, `compress_spike_sources` | Range-compressed spike source representations |
| `mapping_verifier.py` | `verify_soft_core_mapping`, `verify_hardware_config`, `MappingVerificationResult` | Layout mapping verification; hardware config check. `verify_soft_core_mapping` accepts `allow_coalescing` and `hardware_bias` flags that are forwarded to `LayoutIRMapping` so layout-level core counts match actual IR mapping for wide FC layers. With multiple core types, at least one type must fit the largest softcore. `MappingVerificationResult` also carries `host_side_segment_count` (compute-only runs between / around neural segments) and `layout_preview` (compact input/host/latency-group/output flow summary for the wizard). `verify_hardware_config` returns a `"stats"` dict (from `LayoutVerificationStats.to_dict()`) alongside feasibility. Scheduling feasibility is determined entirely by `estimate_passes_for_layout_validated` with typed `pack_layout` validation and fragment expansion — no synthetic fallbacks or override flags. Infeasibility is reported when a softcore's fragments cannot fit any core type. |
| `layout_verification_stats.py` | `LayoutVerificationStats`, `build_layout_verification_stats`, `build_stats_from_packing_result` | Pure stats module: computes total and per-core wasted-axon/neuron percentages, mapped-parameter utilization, `unused_area_total` (rectangular leftover on used cores), `unusable_space_total` and `fragmentation_pct` (strip-shaped internal fragmentation vs used-core capacity), coalescing/splitting counts, and layout-derived neural-segment summaries from a `LayoutPackingResult`. **Chip-level accounting**: when `core_types` is provided, all percentage metrics use chip totals from `core_types` as denominators; `total_hw_cores` equals `sum(ct.count)`. Packing is always performed against real hardware (the partitioner handles fragment expansion), so `len(snaps) <= chip_total_cores` is guaranteed. Idle-core stats are per-type: each idle core's 100% waste / 0% util entry uses its type's actual dimensions. Segment latency summary reports latency groups per neural segment, not absolute global depth indices. UI-agnostic; reusable by wizard, monitor, search, or reports. |
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

**Partitioning (single-pass-per-segment contract)**:
1. Segmentation is entirely the layout mapper's responsibility.  It inserts
   sync barriers (= segment boundaries) wherever the packer cannot place
   the combined cores on the available hardware.  The wizard's neural-segment
   count therefore already reflects the final deployment.
2. The hard-core mapper flushes each segment **as one pass** by calling
   `_flush_neural_segment` against a fresh hardware pool.  No latency-group
   sub-splitting, no bin-packing retry loop, no `_flush_or_split` recursion.
3. If the packer cannot place every core in a segment, the `RuntimeError`
   propagates — this is a loud failure.  The user either up-sizes the
   hardware config or inserts an explicit barrier in the model.  We never
   silently rate-aggregate between latency groups (which would break
   cycle-accurate LIF semantics on the simulator and misrepresent chip
   behaviour).
4. `effective_core_budget()` still reports the 0.8× heterogeneous-discount
   total so the wizard and mapper share one budget number.
5. `estimate_passes_for_layout` / `estimate_passes_for_layout_validated`
   return one entry per `segment_id` and, in the typed variant, validate
   that segment's softcores pack as a whole via `pack_layout`.

**Coalescing constraint**: all coalescing cores for a single NeuralCore (wide
axon tiling) must reside in the **same** pass.  The hardware lacks
membrane-potential initialization across passes, so partial sums cannot be
accumulated temporally.  Both the wizard verifier (fragment expander) and the
build-time mapper (`_validate_coalescing_budget`) enforce this.

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
- `config_schema` imports `mapping.coalescing` for deployment-config validation
- `gui` imports IR types for snapshot extraction

## Exported API (\_\_init\_\_.py)

Core IR types (including `WeightBank`), mapping classes, packing utilities, shared structural helpers (`compute_core_input_count`, `compute_fc_tiling_mode`, `compute_psum_params`), `compute_per_source_scales`, `prune_ir_graph`, and `compute_propagated_pruned_rows_cols`.
`mapping_utils` (the large mapper hierarchy) is intentionally **not** re-exported at
the package level due to its size and star-import patterns; import directly from
`mapping.mapping_utils`.

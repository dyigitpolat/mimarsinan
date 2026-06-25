# pipeline_steps/mapping/ — Mapping Phase

| File | Step class | Gate metric |
|------|------------|-------------|
| `soft_core_mapping_step.py` | `SoftCoreMappingStep` | **Rung 2** — `run_scm_identity_metric` over the 1:1 identity mapping (IR semantics: weights, shifts, banks, segment partition, wire/psum effects). Does **not** cache the packed `hybrid_mapping`. Runs `compute_per_source_scales` before mapping so each layer's upstream activation scale is baked into the deployed effective weights for **every** deployment (idempotent; historically only `WeightQuantizationStep` did this, so the no-weight-quant continuous-ttfs deploy shipped weights missing the input-scale factor and tripped the NF↔SCM gate). After IR pruning it runs `_run_onchip_majority_gate` (`mapping.verification.onchip_majority.assert_onchip_majority_or_raise`): the tiered-validity FLOOR gate (params-based defense-in-depth) raises `OnchipMajorityError` only BELOW the floor; between the floor and the 50% majority a mapping is VALID_FLAGGED and deploys (config `onchip_majority_gate` default-on, floor `onchip_majority_min_fraction`=0.2). It then runs `_run_capacity_gate` (`mapping.verification.capacity.estimate_cores_needed`) BEFORE HardCoreMapping: the E4 placement-capacity diagnostic raises `CapacityExceededError` (naming the overflowing segment + `cores_needed`/`cores_available`) when the IR provably overflows the declared core budget, turning the late greedy-packer crash (`"No more hard cores available"`) into an early diagnosable verdict (config `capacity_gate` default-on). The gate is scheduling-aware: `allow_scheduling` is resolved from `platform_constraints` (the chip's `MappingStrategy` capability), and when scheduling is permitted the gate does NOT raise if the PEAK reprogram phase fits — only when the peak phase or a single atomic coalescing bundle overflows the whole budget (so VGG16@224 reads feasible-via-scheduling instead of infeasible). |
| `core_quantization_verification_step.py` | `CoreQuantizationVerificationStep` | — |
| `hard_core_mapping_step.py` | `HardCoreMappingStep` | **Rung 3** — `run_hcm_mapping_metric` over the packed mapping (packing: placement, padding, reindex, coalescing, splitting, scheduling). |

Uses `pipelining.hybrid_mapping_consumer` and `pipelining.simulation_factory` for the gate metrics and cached hybrid mappings. HCM builds the packed mapping via `load_hybrid_mapping_for_step` when uncached. The old `build_spiking_flow_for_metric` alias is gone.

## Helper modules (consumed by `SoftCoreMappingStep`)

| File | Export | Purpose |
|------|--------|---------|
| `fused_linear.py` | `FusedLinear` | Bias-folded linear; `bring_back_bias` restores a plain `nn.Linear` before mapping. |
| `soft_core_mapping_ir_pruning.py` | `apply_ir_pruning_if_enabled` | In-loop mask pruning: compacts already-zeroed rows/cols on the IR when `plan.pruning` is on. |
| `soft_core_structured_pruning.py` | `apply_structured_pruning_if_enabled` | **D4** structured pre-mapping pruning: when `prune_sparsity > 0` (default `0.0` ⇒ no-op, byte-identical), runs `transformations.pruning.magnitude.prune_perceptron_chain` on the fused model BEFORE mapping, structurally removing low-magnitude output channels so the IR maps to fewer cores / reprogram phases (see `docs/research/findings/D4_pruning_scheduling_cost.md`). Runs after norm-fusion (`normalization == Identity`). |
| `soft_core_mapping_viz.py` | `write_ir_graph_visualizations` | Soft-core IR heatmaps / flowchart artifacts. |

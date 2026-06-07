# pipeline_steps/mapping/ — Mapping Phase

| File | Step class | Gate metric |
|------|------------|-------------|
| `soft_core_mapping_step.py` | `SoftCoreMappingStep` | **Rung 2** — `run_scm_identity_metric` over the 1:1 identity mapping (IR semantics: weights, shifts, banks, segment partition, wire/psum effects). Does **not** cache the packed `hybrid_mapping`. |
| `core_quantization_verification_step.py` | `CoreQuantizationVerificationStep` | — |
| `hard_core_mapping_step.py` | `HardCoreMappingStep` | **Rung 3** — `run_hcm_mapping_metric` over the packed mapping (packing: placement, padding, reindex, coalescing, splitting, scheduling). |

Uses `pipelining.hybrid_mapping_consumer` and `pipelining.simulation_factory` for the gate metrics and cached hybrid mappings. HCM builds the packed mapping via `load_hybrid_mapping_for_step` when uncached. The old `build_spiking_flow_for_metric` alias is gone.

# mapping/verification/ — Layout Verification and HW Suggestions

Wizard and pipeline layout verification, HW config suggestions, and mapping verifier.

## Key modules

| File | Symbols | Role |
|------|---------|------|
| `layout_mapping_service.py` | Layout service API | Orchestrates layout verify flow |
| `layout_verification_stats.py` | Stats collection | Layout verification metrics |
| `layout_request.py` | Request types | Wizard layout request payloads |
| `mapping_verifier.py` | `MappingVerifier` | End-to-end mapping verification |
| `hw_config_suggester.py` | HW suggestions | Suggests platform constraints from layout |
| `wizard_layout_verify.py` | Wizard entry | GUI/wizard layout verify |
| `onchip_majority.py` | `assert_onchip_majority_or_raise`, `compute_onchip_fraction`, `count_host_params`, `OnchipParamBreakdown`, `OnchipMajorityError` | Tiered-validity FLOOR gate (params-based defense-in-depth). On-chip = `total_params - host`; host = unique ComputeOp `module.parameters()` (deduped by identity; bound tensors only when no module). Raises `OnchipMajorityError` only when `onchip_fraction < FLOOR` (default `min_fraction=0.2`); between the floor and the 50% majority a mapping is VALID_FLAGGED and must NOT raise (config `onchip_majority_gate` default-on, floor `onchip_majority_min_fraction` default 0.2). |
| `onchip_fraction.py` | `estimate_onchip_fraction`, `assert_onchip_majority_estimate_or_raise`, `OnchipFractionEstimate`, `classify_validity`, `ValidityVerdict` | STATIC pre-check from the model SPEC alone (no training, no core placement, no `generated/` run). Converts via `convert_torch_model`, classifies host = ComputeOpMapper modules + `is_encoding_layer` perceptrons, on-chip = remaining perceptrons. `metric="params"` reproduces `count_host_params` exactly; `metric="macs"` reports the on-chip forward-compute fraction (attention is param-light but MAC-heavy) by hooking a single flow forward. **`classify_validity`** is the TIERED gate v2 (one flow, both metrics): `INVALID` iff `min(param_frac, mac_frac) < floor` (default 0.20); `VALID` iff both `>= majority` (default 0.50); `VALID_FLAGGED` otherwise. It also classifies each host op into `research_gap_ops` (no on-chip SNN mapping yet — MultiheadAttention/LayerNorm/GELU, the research frontier) vs `placement_fixable_ops` (supported encoders host-placed under `subsume`, fixable via `offload`). |
| `capacity/` (subpackage) | `estimate_cores_needed`, `CapacityEstimate`, `CapacityExceededError` | **E4 placement-capacity diagnostic** (see `capacity/ARCHITECTURE.md`). STATIC, SOUND lower bound on the hard cores an `IRGraph` needs on a core budget — WITHOUT running the greedy placer — so a provably-infeasible config is rejected EARLY (a `CapacityExceededError` naming `cores_needed`/`cores_available`/`overflowing_segment`) instead of crashing late inside the packer with `"No more hard cores available"` (E3: VGG16@224 needs ~316k cores on a 1000-core budget). Partitions the IR into neural segments (`layout.segmentation.partition_ir_graph`); each segment's bound mirrors the diagonal packer = `max(ceil(Σ axons/max_axons), ceil(Σ neurons/max_neurons), max per-core frags·groups)` (`coalescing_fragment_count` × `ceil(out/max_neurons)`); sums across segments (one core pool) and names the first cumulative overflow. Because it is a lower bound it NEVER rejects a config the packer could place (VGG16@32 → 830 ≤ 2048; lenet5 → 40 ≤ 120, actual placed 57). `CapacityEstimate.raise_if_infeasible()` is the gate. Re-exported by `mapping.verification`. |

## Exported API (`__init__.py`)

`OnchipMajorityError`, `OnchipParamBreakdown`, `assert_onchip_majority_or_raise`, `compute_onchip_fraction`, `count_host_params`, `OnchipFractionEstimate`, `ValidityVerdict`, `assert_onchip_majority_estimate_or_raise`, `classify_validity`, `estimate_onchip_fraction`, `CapacityEstimate`, `CapacityExceededError`, `estimate_cores_needed`.

## Dependents

- `gui.server`, `search` problems, pipeline soft/hard core mapping steps.
- `SoftCoreMappingStep` runs the tiered-validity FLOOR gate after IR pruning,
  before persisting the `ir_graph` (raises only below the 0.2 floor). It then runs
  the E4 `estimate_cores_needed` capacity gate (`_run_capacity_gate`) BEFORE
  HardCoreMapping: raises `CapacityExceededError` when the IR provably overflows the
  declared core budget (config `capacity_gate` default-on), replacing the late
  greedy-packer crash with an early diagnosable verdict.
- `scripts/campaign/scheduler.py` runs `classify_validity` as an enqueue pre-check:
  it rejects only INVALID jobs (`min(param,mac) < floor`, no GPU claimed) and admits
  VALID + VALID_FLAGGED, logging the named research gap for flagged jobs. Config
  floor `deployment_parameters.onchip_min_fraction` (0.20), majority
  `onchip_majority_fraction` (0.50). It also runs `estimate_cores_needed`
  (`capacity_precheck`): rejects a config whose static lower bound exceeds its core
  budget (no GPU claimed), opt-out `deployment_parameters.capacity_gate`; estimate
  failures are NON-FATAL (enqueued anyway).

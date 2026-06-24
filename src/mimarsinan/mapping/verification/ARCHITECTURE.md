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
| `onchip_majority.py` | `assert_onchip_majority_or_raise`, `compute_onchip_fraction`, `count_host_params`, `OnchipParamBreakdown`, `OnchipMajorityError` | Hard validity gate: the deployed parameter MAJORITY must be on chip cores. On-chip = `total_params - host`; host = unique ComputeOp `module.parameters()` (deduped by identity; bound tensors only when no module). Raises `OnchipMajorityError` when `onchip_fraction < 0.5` (config `onchip_majority_gate` default-on, floor `onchip_majority_min_fraction`). |
| `onchip_fraction.py` | `estimate_onchip_fraction`, `assert_onchip_majority_estimate_or_raise`, `OnchipFractionEstimate` | STATIC pre-check of the same gate from the model SPEC alone (no training, no core placement, no `generated/` run). Converts via `convert_torch_model`, classifies host = ComputeOpMapper modules + `is_encoding_layer` perceptrons, on-chip = remaining perceptrons. `metric="params"` reproduces `count_host_params` exactly; `metric="macs"` reports the on-chip forward-compute fraction (attention is param-light but MAC-heavy) by hooking a single flow forward. Dependency-light enough to import in the campaign scheduler. |

## Exported API (`__init__.py`)

`OnchipMajorityError`, `OnchipParamBreakdown`, `assert_onchip_majority_or_raise`, `compute_onchip_fraction`, `count_host_params`, `OnchipFractionEstimate`, `assert_onchip_majority_estimate_or_raise`, `estimate_onchip_fraction`.

## Dependents

- `gui.server`, `search` problems, pipeline soft/hard core mapping steps.
- `SoftCoreMappingStep` runs the on-chip-majority gate after IR pruning,
  before persisting the `ir_graph`.
- `scripts/campaign/scheduler.py` runs `estimate_onchip_fraction` as an enqueue
  pre-check so a host-majority job never claims a GPU.

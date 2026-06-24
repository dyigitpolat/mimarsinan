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

## Exported API (`__init__.py`)

`OnchipMajorityError`, `OnchipParamBreakdown`, `assert_onchip_majority_or_raise`, `compute_onchip_fraction`, `count_host_params`.

## Dependents

- `gui.server`, `search` problems, pipeline soft/hard core mapping steps.
- `SoftCoreMappingStep` runs the on-chip-majority gate after IR pruning,
  before persisting the `ir_graph`.

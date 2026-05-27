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

## Dependents

- `gui.server`, `search` problems, pipeline soft/hard core mapping steps.

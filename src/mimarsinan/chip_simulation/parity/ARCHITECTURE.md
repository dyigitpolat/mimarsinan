# chip_simulation/parity/ — Cross-Backend Record Parity

Compares simulation run records between backends (HCM reference vs SANA-FE / TTFS).

| File | Symbols | Role |
|------|---------|------|
| `record_compare.py` | `FieldDiff`, `compare_segment_records`, `format_first_diff` | Structural diff of per-segment records |

## Dependents

- `pipelining.pipeline_steps.verification.sanafe_simulation_step`, integration parity tests.

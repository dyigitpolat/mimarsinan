# mapping/export/ — Chip Export and Quantization

Exports mapped models for chip deployment and verifies quantized IR.

## Key modules

| File | Symbols | Role |
|------|---------|------|
| `chip_export.py` | Export routines | C++/chip artifact export |
| `chip_quantize.py` | `verify_ir_graph_quantized` | Post-quantization IR verification |

## Dependents

- `pipelining.pipeline_steps.mapping.core_quantization_verification_step`, code generation.

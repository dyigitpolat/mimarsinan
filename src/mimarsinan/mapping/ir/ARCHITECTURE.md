# mapping/ir/ — Intermediate Representation

Unified graph model for neural cores, compute ops, and weight banks.

## Key modules

| File | Symbols | Role |
|------|---------|------|
| `types.py` | `IRSource`, `IRNode`, `NeuralCore`, `ComputeOp`, `WeightBank` | Core datatypes and `IRSource` sentinel conventions |
| `graph.py` | `IRGraph` | Graph container, execution helpers, output sources. `build_token` (uuid per construction; `None` on legacy pickles) — provenance stamp so derived artifacts (the packed hybrid mapping) can be detected as stale across pipeline resumes. |
| `legacy_convert.py` | SoftCore conversion helpers | Bridge legacy SoftCore layouts to IR nodes |

## Dependencies

- **Internal**: `mapping.spike_source_spans`, `mapping.activation_scales`
- **External**: `numpy`, `torch` (weight material)

## Dependents

- `mapping.ir_mapping`, `mapping.mappers`, `mapping.packing.hybrid_hardcore_mapping`, `chip_simulation`, pipeline mapping steps.

## Invariants

- Off-sources use `node_id=-1`; chip input uses `-2`; const-on uses `-3`.
- `ComputeOp.execute_on_gathered` receives flat `(B, N)` tensors.

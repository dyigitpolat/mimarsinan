# mapping/latency/ — Latency and Upstream Closure

Computes per-node latency tiers on IR and chip mappings.

## Key modules

| File | Symbols | Role |
|------|---------|------|
| `ir.py` | `IRLatency` | Topological latency on `IRGraph` |
| `chip.py` | `ChipLatency` | Cycle scheduling on `HardCoreMapping` |
| `upstream.py` | `iter_upstream_neural_ids` | Shared upstream neural ID closure |

## Dependencies

- **Internal**: `mapping.ir`, `mapping.packing.softcore_mapping`
- **External**: none

## Dependents

- `mapping.ir_mapping`, `models.hybrid_core_flow`, SANA-FE runner.

## Invariants

- `IRLatency` and `ChipLatency` remain separate facades (different inputs and pipeline phases).
- See [`LATENCY.md`](LATENCY.md) for firing-mode boundaries.

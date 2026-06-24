# mapping/ — Model-to-Hardware Mapping

Converts PyTorch models to an intermediate representation (IR) and packs the IR into physical hardware cores.

## Subpackages

| Directory | Doc | Role |
|-----------|-----|------|
| `ir/` | [ir/ARCHITECTURE.md](ir/ARCHITECTURE.md) | `IRGraph`, `NeuralCore`, `ComputeOp` |
| `pruning/` | [pruning/ARCHITECTURE.md](pruning/ARCHITECTURE.md) | IR pruning, liveness, segmentation |
| `packing/` | [packing/ARCHITECTURE.md](packing/ARCHITECTURE.md) | Soft/hard cores, bin packing, hybrid mapping |
| `verification/` | [verification/ARCHITECTURE.md](verification/ARCHITECTURE.md) | Layout verifier, HW suggester |
| `latency/` | [latency/ARCHITECTURE.md](latency/ARCHITECTURE.md) | `IRLatency`, `ChipLatency`, upstream closure |
| `platform/` | [platform/ARCHITECTURE.md](platform/ARCHITECTURE.md) | Platform constraints, tiling structure |
| `export/` | [export/ARCHITECTURE.md](export/ARCHITECTURE.md) | Chip export and IR quantization verify |
| `layout/` | [layout/ARCHITECTURE.md](layout/ARCHITECTURE.md) | Shape-only layout SSoT |
| `mappers/` | [mappers/ARCHITECTURE.md](mappers/ARCHITECTURE.md) | Mapper hierarchy |
| `support/` | [support/ARCHITECTURE.md](support/ARCHITECTURE.md) | Scales, bias bakes (negative-value shift, TTFS), geometry, scheduling |

## Root modules (orchestration)

| File | Role |
|------|------|
| `ir_mapping.py` | `IRMapping` — materializes weights into `IRGraph` |
| `mapping_utils.py` | Mapper registration and graph utilities |
| `weight_reuse.py` | Time-domain weight-reuse phase classification (round-1 keystone, default-off): `classify_segment_phases` / `weight_reuse_plan_from_graph` group a segment's `NeuralCore`s by `weight_bank_id` into N reprogram + M reuse passes (`SegmentReusePhases` / `WeightReusePlan`); `format_weight_reuse_summary` is the SCM-gate one-liner. Pure read of the IR; gated by `ChipCapabilities.allow_weight_reuse`. |
| `model_representation.py` | Dual-purpose mapper graph.  `__call__` is a memory-frugal topological executor: a reverse-dependency refcount (computed once in `_ensure_exec_graph` as `self._consumer_count`) drives `del values[dep]` once every consumer has run, so peak live values is bounded by the maximum simultaneously-live working set rather than the total node count.  `self._peak_live_values` is exposed for testing / memory introspection. |

## Multi-input `ComputeOpMapper` contract

A `ComputeOpMapper` whose source list has length > 1 receives a tuple of
tensors at forward time (one per source, in source order).  Two guards are
in force before the wrapped module is invoked:

1. **Compile-time shape recording.**  `MapperGraphConverter` records one
   batch-stripped shape per source in `ComputeOpMapper.input_shapes`
   (`tuple[tuple[int, ...] | None, ...]`).  Missing FX metadata becomes
   `None` for that source; downstream callers fall back to source-array
   shapes.
2. **Runtime broadcast guard.**  `ComputeOpMapper._check_broadcastable`
   runs `torch.broadcast_shapes` on the actual input shapes before
   `self.module(*inputs, ...)` is called.  Any broadcast failure raises
   `ShapeMismatchError` with the op name, observed shapes, and recorded
   `input_shapes` — no large tensor is ever allocated with malformed
   shapes.

Import from subpackages directly (e.g. `from mimarsinan.mapping.ir import IRGraph`). [`__init__.py`](__init__.py) re-exports the public mapping API.

Run `python scripts/import_path_inventory.py --check-legacy-path` to audit banned legacy paths.

# mapping/ — PyTorch model → IR → packed hardware cores

Takes the mapper graph produced by `torch_mapping` (a `ModelRepresentation` of
`Mapper` nodes) and lowers it to deployable hardware: `IRMapping` materializes
weights into an `IRGraph` of `NeuralCore` / `ComputeOp` / `WeightBank` nodes,
pruning compacts and segments that graph, and packing bins the resulting soft
cores into `HardCore`s under `ChipCapabilities` constraints (flat or hybrid
multi-stage). Latency scheduling, shape-only layout estimation (for search),
layout verification, and chip export round out the path from trained model to
simulator-ready chip description. Design notes live beside the code in
`FIRING.md` (firing-mode contract) and `LATENCY.md` (which latency engine to
use when).

## Key files
| File | Purpose |
|---|---|
| `ir_mapping_class.py` | `IRMapping` — the unified full-weight mapper (emit mixin over the core base) |
| `ir_mapping_class_base.py` | `IRMappingCore` — mapping walk producing an `IRGraph`, built on the shape-only `LayoutIRMapping` tiling base |
| `ir_mapping_class_emit.py` | `IRMappingEmitMixin` — emits concrete `NeuralCore` nodes with materialized weights alongside the shape walk |
| `map_model_to_ir.py` | `map_model_to_ir` — one-call convenience wrapper around `IRMapping.map` |
| `model_representation.py` | `ModelRepresentation` — mapper-graph DAG with a memory-frugal refcounted topological executor and perceptron enumeration |
| `mapping_utils.py` | Legacy star re-export facade; import from the concrete modules in new code |
| `weight_reuse.py` | Time-domain weight-reuse phase classification of segments by `weight_bank_id` (default-off, pure read of the IR) |
| `ir/` | Unified IR: `IRGraph` container, node types (`NeuralCore`, `ComputeOp`, `WeightBank`, `IRSource`), legacy conversions |
| `mappers/` | Mapper hierarchy consumed by the graph: base, structural, perceptron, leading-dim, conv1d/conv2d, scale propagation |
| `layout/` | Shape-only layout SSoT (`LayoutIRMapping`, layout plan/packer, segmentation) for fast architecture-search estimation |
| `platform/` | `ChipCapabilities`, `MappingStrategy`, platform constraints, tiling/coalescing structure |
| `pruning/` | IR pruning, liveness semantics, mask/compaction application, boundary policy, graph segmentation |
| `packing/` | `SoftCore`/`HardCore` bin packing, placement engine, hybrid multi-stage mapping (`HybridHardCoreMapping`) |
| `latency/` | `IRLatency` (IR topology tiers) and `ChipLatency` (packed-chip cycle scheduling) plus upstream closure |
| `support/` | Shared mechanisms: activation/per-source scales, bias compensation, core geometry, source spans, residual merge, scheduling |
| `verification/` | Layout verification services, capacity checks, hardware suggester, on-chip fraction/majority metrics, wizard verify |
| `export/` | Chip export (`cpp_chip_model` emission) and IR quantization verify for simulation |
| `onchip_attention/` | D5 research frontier: attention/LayerNorm on-chip mappability verdicts; not in the deployment path |

## Dependencies
- `code_generation` — `cpp_chip_model` chip types (`SpikeSource`, chip model) for export, softcore packing, and spike-source spans.
- `models` — `Perceptron` and nn layers consumed by mappers; builders registry for the wizard layout verify.
- `transformations` — `PerceptronTransformer` weight/bias extraction; weight quantization and quantization bounds/verify for export; `pruning.committed_masks.commit_layer_pruning` in the conv mappers' forward (F.conv bypasses layer hooks).
- `torch_mapping` — `convert_torch_model` and encoding-layer marking in the wizard layout verify.
- `pipelining` — `ModelRegistry` model loading in the wizard layout verify.
- `chip_simulation` — spiking-semantics constants for pruning liveness.
- `tuning` — activation-shift calculation for bias compensation.
- `common` — env flags (`cuda_debug_enabled`) and `best_effort` wrapper.

## Dependents
- `chip_simulation` — runs simulations from mapped `HardCoreMapping`s / segments.
- `code_generation` — generates chip code from mapping outputs.
- `config_schema` — mapping-related configuration surfaces.
- `gui` — visual inspection of mappings.
- `models` — model-side hooks into mapping types.
- `pipelining` — pipeline steps that drive mapping, packing, and verification.
- `search` — architecture search over shape-only layout estimates.
- `spiking` — spiking-node interplay with IR types.
- `torch_mapping` — builds the mapper graph that this module consumes.
- `transformations` — transformations parameterized by mapping structures.
- `tuning` — tuning stages that consult mapping/latency info.
- `visualization` — plots of IR graphs and core layouts.

## Exported API
`__init__.py` re-exports the public mapping surface:
- IR types: `IRSource`, `IRNode`, `NeuralCore`, `ComputeOp`, `IRGraph`, `WeightBank`.
- Mapping: `IRMapping` (model → IR).
- Platform: `ChipCapabilities`, `MappingStrategy`, `compute_core_input_count`, `compute_fc_tiling_mode`.
- Packing: `SoftCore`, `HardCore`, `HardCoreMapping`, `greedy_pack_softcores`; hybrid — `SegmentIOSlice`, `HybridStage`, `HybridHardCoreMapping`, `build_hybrid_hard_core_mapping`.
- Latency: `ChipLatency`, `IRLatency`.
- Scales: `compute_per_source_scales`.
- Pruning: `prune_ir_graph`, `compute_propagated_pruned_rows_cols`, `GlobalPruningResult`, `compute_global_pruned_sets`, `compute_model_io_boundary_policy`.

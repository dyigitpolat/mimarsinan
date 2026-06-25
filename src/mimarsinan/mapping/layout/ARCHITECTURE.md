# mapping/layout/ — Shape-Only Mapping (Single Source of Truth)

`LayoutIRMapping` is the single source of truth for every mapping
decision: tiling mode (`single` / `output_tiled` / `coalescing`),
bias-axon counting, shared-bank wiring,
latency / segment / threshold-group assignment. A wide fan-in maps via
`coalescing` (one full-width core the packer fuses from N hard cores into a wider
crossbar — bit-exact) when the chip supports inter-core membrane transfer
(`allow_coalescing`); otherwise it is unmappable and raises
`WideFanInUnsupportedError`. The lossy firing partial-sum (spike-domain) fallback
was removed. Coalescing (axon overflow) and neuron splitting (fan-out) are
independent chip-capability flags.

`onchip_residual_merge` (default off) is a deployment-mode flag: when set,
`LayoutIRMapping.map` first lowers every param-free equal-width residual add
(`mapping/support/residual_merge.py`) onto the crossbar as a signed-IF
identity-merge core (Tier-1), keeping the residual on-chip in one segment instead
of a host ComputeOp add (Tier-0). Off → the host add, byte-identical. Tier-1 is a
VALID, characterized deployment, NOT bit-exact to Tier-0 (a bounded ~1/T in-segment
IF re-quant; see `docs/research/findings/D2_tier1_deployable.md`).

The real `mapping.ir_mapping.IRMapping` is a subclass that overrides
the emission hooks (`add_neural_core`, `add_shared_neural_core`,
`add_compute_op`, `register_weight_bank`) to additionally attach weight
material and build an `IRGraph`.  Both paths therefore emit byte-identical
softcore shapes for the same input model — the wizard / architecture-search
flow stops at the shape-only layer while the deployment pipeline continues
to materialise weights.

## Threshold groups

`threshold_group_id = perceptron_index`: every softcore produced by a
single perceptron (output-tiles, shared-bank positions)
shares one integer id, and the packer treats group-id equality as the
sharing-compatibility rule.  Softcores without a `perceptron_index`
(e.g. synthesised accumulator cores) fall back to a unique negative id
(never collides with a non-negative perceptron index).

This mirrors the real pipeline's behaviour: shared-weight cores end up
with identical quantization scales (→ identical float thresholds), so
"same perceptron" is a byte-accurate proxy for "same threshold class"
without requiring trained weights.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `layout_types.py` | `LayoutSoftCoreSpec`, `LayoutHardCoreType`, `LayoutHardCoreInstance`, `LayoutCoreSnapshot`, `LayoutPackingResult` | Data classes for layout-only core specifications and packing results. |
| `layout_ir_mapping.py` | `LayoutIRMapping` | Shape-only mapping backend (core state, `map`, neural emission hooks) |
| `layout_ir_mapping_fc.py` | `_LayoutIRMappingFC` | FC tiling dispatch (`map_fc`: single / coalescing / output-tiled) |
| `layout_ir_mapping_finalize.py` | `_LayoutIRMappingFinalize` | Latency tags, segment ids, threshold groups, layout preview (delegates to `segmentation.py`) |
| `segmentation.py` | `partition_ir_graph`, `compute_segment_ids`, `compute_node_latencies`, `compute_host_side_segment_count`, `NeuralSegment`, `HostSegment` | Single source of neural/host segmentation: the ordered partition the HCM builders flush **and** the dependency-graph segment ids/latencies the layout finalizer uses. |
| `layout_source_view.py` | `LayoutSourceView`, `concat_source_views`, `stack_source_views`, `node_ids_of`, `total_size` | Lightweight composable shape descriptor that duck-types as a numpy object array of `IRSource` for the mapper graph; defers per-cell `IRSource` allocation until forced via `np.asarray`.  See "LayoutSourceView contract" below. |
| `layout_packer.py` | `pack_layout`, `LayoutMaterializer` | Shape-only `Materializer` + thin wrapper over the shared `placement_engine.run_placement`. |
| `softcore_spec_adapter.py` | `spec_from_neural_core`, `spec_from_softcore` | Derive a shape-only `LayoutSoftCoreSpec` from an IR `NeuralCore` (scheduled splitting) or a compacted runtime `SoftCore` (deployment plan). |
| `layout_plan.py` | `LayoutPlan`, `build_layout_plan` | Single placement-plan + stats artifact. `build_layout_plan` feeds the wizard/NAS/snapshot "planned" path; `LayoutPlan.from_hybrid_mapping` derives the same `LayoutVerificationStats` from a compiled deployment mapping, so miniview and deployment stats share one engine. |

## LayoutSourceView contract

`LayoutIRMapping`'s emission hooks return `LayoutSourceView` rather than a
materialised numpy array of `IRSource`.  The view exposes the duck-type
surface the mapper graph uses (`.shape`, `.ndim`, `.size`, `.dtype`,
`.flatten()`, `.reshape()`, `.transpose()`, `__getitem__`, `__len__`,
`__iter__`, `__array__`) without ever allocating per-cell `IRSource`
instances; ops the view does not natively cover (numpy `transpose` of an
unusual shape, einops, pad, moveaxis) trigger materialisation via
`__array__`.  Concat / stack mapper sites use `concat_source_views` /
`stack_source_views` so view-only chains stay free of materialisation.

Invariants:

- `node_ids_of(view)` returns the set of producer node ids feeding the
  view -- no per-cell scan.  `total_size(view)` likewise reads the size
  metadata directly.
- The full `IRMapping` subclass materialises each emission boundary via
  `_convert_sources` and re-wraps its own return value with
  `np.asarray(..., dtype=object)`, so downstream consumers in the
  IRMapping path see only real numpy arrays of `IRSource` -- the view
  never leaks into the IR graph or any simulator.
- Materialisation of a from-producer view yields
  `IRSource(producer_node_id, original_flat_index)` for every cell,
  preserving the connectivity semantics the IR graph expects.

The shape-only wizard / NAS path stays view-only end to end -- the
single source of truth, lightweight as intended.

## Dependencies

- **Internal**: `mapping.ir` (`IRSource`), `mapping.mapping_structure`
  (`compute_core_input_count`, `compute_fc_tiling_mode`), `mapping.core_packing`
  (`greedy_pack_softcores`).
- **External**: `numpy`.

## Dependents

- `mapping.ir_mapping.IRMapping` — subclasses `LayoutIRMapping`.
- `mapping.mapping_verifier.verify_soft_core_mapping` — wizard-facing
  feasibility check.
- `search.problems.joint_arch_hw_problem` — hw-aware architecture search.

## Exported API (`__init__.py`)

All layout types, `LayoutIRMapping`, `pack_layout`, `LayoutPlan` /
`build_layout_plan`, the segmentation API (`partition_ir_graph`,
`compute_segment_ids`, `compute_node_latencies`,
`compute_host_side_segment_count`, `NeuralSegment`, `HostSegment`, `Segment`),
and the spec adapters (`spec_from_neural_core`, `spec_from_softcore`).

# mapping/layout/ — Shape-Only Mapping (Single Source of Truth)

`LayoutIRMapping` is the single source of truth for every mapping
decision: tiling mode (`single` / `output_tiled` / `psum` / `coalescing`),
bias-axon counting, psum decomposition parameters, shared-bank wiring,
latency / segment / threshold-group assignment.

The real `mapping.ir_mapping.IRMapping` is a subclass that overrides
the emission hooks (`add_neural_core`, `add_shared_neural_core`,
`add_compute_op`, `register_weight_bank`) to additionally attach weight
material and build an `IRGraph`.  Both paths therefore emit byte-identical
softcore shapes for the same input model — the wizard / architecture-search
flow stops at the shape-only layer while the deployment pipeline continues
to materialise weights.

## Threshold groups

`threshold_group_id = perceptron_index`: every softcore produced by a
single perceptron (output-tiles, psum fragments, shared-bank positions)
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
| `layout_ir_mapping.py` | `LayoutIRMapping` | Shape-only mapping backend shared by the wizard, architecture search, and the real `IRMapping` (as its base class).  Owns every structural decision. |
| `layout_packer.py` | `pack_layout` | Wraps `greedy_pack_softcores` around layout types. |

## Dependencies

- **Internal**: `mapping.ir` (`IRSource`), `mapping.mapping_structure`
  (`compute_core_input_count`, `compute_fc_tiling_mode`,
  `compute_psum_params`), `mapping.core_packing`
  (`greedy_pack_softcores`).
- **External**: `numpy`.

## Dependents

- `mapping.ir_mapping.IRMapping` — subclasses `LayoutIRMapping`.
- `mapping.mapping_verifier.verify_soft_core_mapping` — wizard-facing
  feasibility check.
- `search.problems.joint_arch_hw_problem` — hw-aware architecture search.

## Exported API (`__init__.py`)

All layout types, `LayoutIRMapping`, and `pack_layout`.

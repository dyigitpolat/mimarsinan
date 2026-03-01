# mapping/ -- Model-to-Hardware Mapping

Converts PyTorch models to an intermediate representation (IR) and then packs
the IR into physical hardware cores.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `ir.py` | `IRSource`, `IRNode`, `NeuralCore`, `ComputeOp`, `IRGraph`, `WeightBank` | Unified IR: directed graph of neural cores and compute operations. `WeightBank` stores shared weights for conv-style layers; `NeuralCore` can either own its `core_matrix` or reference a bank via `weight_bank_id`. |
| `ir_mapping.py` | `IRMapping` | Converts `ModelRepresentation` (mapper graph) to `IRGraph`; handles output/axon tiling. Provides `register_weight_bank()` and `add_shared_neural_core()` for conv-style shared-weight mapping. |
| `mapping_utils.py` | `Mapper`, `ModelRepresentation`, `SoftCoreMapping`, `PerceptronMapper`, `Conv2DPerceptronMapper`, ... | Mapper hierarchy: dual-purpose DAG for forward pass and hardware mapping |
| `softcore_mapping.py` | `SoftCore`, `HardCore`, `HardCoreMapping` | Logical-to-physical core representations and packing |
| `core_packing.py` | `greedy_pack_softcores` | Generic best-fit greedy bin-packing algorithm |
| `hybrid_hardcore_mapping.py` | `HybridHardCoreMapping`, `HybridStage`, `SegmentIOSlice`, `build_hybrid_hard_core_mapping` | Multi-segment deployable program with state-buffer I/O |
| `chip_latency.py` | `ChipLatency` | Calculates chip simulation latency from core graph |
| `ir_latency.py` | `IRLatency` | Computes per-node latency tiers in the IR graph |
| `per_source_scales.py` | `compute_per_source_scales` | Traverses mapper graph to set per-input-channel `per_input_scales` on each perceptron; handles branching (concat) and dimension-rearranging mappers (falls back to mean when channel counts don't align) |
| `ir_source_spans.py` | `IRSourceSpan`, `compress_ir_sources` | Range-compressed IR source representations for efficient simulation |
| `spike_source_spans.py` | `SpikeSourceSpan`, `compress_spike_sources` | Range-compressed spike source representations |

### Subdirectory

| Directory | Purpose |
|-----------|---------|
| `layout/` | Shape-only layout estimation for architecture search (no weights) |

## Dependencies

- **Internal**: `code_generation.cpp_chip_model` (`SpikeSource`), `models.layers`, `models.perceptron_mixer.perceptron` (`Perceptron`), `transformations` (`PerceptronTransformer`, `TensorQuantization`).
- **External**: `torch`, `numpy`, `einops`.

## Dependents

- `models` imports `Mapper` classes for building mapper graphs, IR types for spiking simulation
- `pipelining` imports `IRMapping`, `IRGraph`, `NeuralCore` for mapping steps
- `chip_simulation` imports mappings for nevresim code generation
- `visualization` imports IR and mapping types for DOT generation
- `tuning` imports `NeuralCore` for CoreFlow tuning
- `search` imports layout utilities for architecture search
- `gui` imports IR types for snapshot extraction

## Exported API (\_\_init\_\_.py)

Core IR types (including `WeightBank`), mapping classes, packing utilities, and `compute_per_source_scales`.
`mapping_utils` (the large mapper hierarchy) is intentionally **not** re-exported at
the package level due to its size and star-import patterns; import directly from
`mapping.mapping_utils`.

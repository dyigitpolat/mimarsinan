# torch_mapping/ — native PyTorch → mimarsinan Mapper DAG conversion

Converts a trained native `nn.Module` into a mimarsinan `ModelRepresentation`
(Mapper DAG) wrapped in a `ConvertedModelFlow`, so pretraining runs on the
unmodified torch model and the adaptation/quantization/mapping stages operate
on Perceptrons. The pipeline is: FX-trace with shape propagation → graph
normalization (Linear fusion) → representability analysis with a BN/activation
absorption plan → node-by-node mapper emission. MM→BN?→ACT chains become
Perceptron mappers (on-chip candidates); shape-only ops fold into structural
mappers; every other op flows through one generic host `ComputeOpMapper` path.

## Key files
| File | Purpose |
|---|---|
| `conversion_probe.py` | Strict-by-default warmup forward (`probe_forward`, `ProbeResult`, `ConversionProbeError`); parses the ModelRepresentation failure marker to name the offending mapper node |
| `converted_model_flow.py` | `ConvertedModelFlow`: PerceptronFlow subclass wrapping the converted `ModelRepresentation`; registers all perceptrons and graph-node modules so `.to()`/`state_dict()`/`parameters()` reach every one |
| `converter.py` | Public facade: `check_representability` and `convert_torch_model` (trace → normalize → analyze → convert → mark encoding layers → probe); `strict=True` default, `encoding_layer_placement="subsume"/"offload"` |
| `converter_handlers/` | Mixins for `MapperGraphConverter`: `LinearConvertMixin`/`ConvConvertMixin` package Perceptrons with absorbed BN/activation (string conv padding rejected), `StructuralConvertMixin` holds cat/flatten shortcuts and absorption helpers, `converter_contract.py` is the typing-only host contract |
| `encoding_layers.py` | `mark_encoding_layers` (subsume marks segment-start perceptrons as host spike-train ComputeOps; offload clears the mark so they map on-chip), `encoder_deploys_as_staircase_hop` (placement→arithmetic SSOT: subsumed encoders run their own staircased module and take the sync entry half-step), and `segment_entry_perceptrons` (first on-chip perceptron per neural segment — the TTFS wire-contract seam) |
| `fx_shape_utils.py` | Shared FX `tensor_meta` extraction (`node_input_shapes`, `node_output_shape`, `strip_batch`) plus `fx_literal_int` literal coercion and `node_target_str` |
| `graph_normalization.py` | `normalize_fx_graph`: in-place fusion of consecutive Linears (walking through Identity/BN and folding BN into the preceding Linear) followed by dead-code elimination |
| `mapper_graph_converter.py` | `MapperGraphConverter`: walks the FX graph emitting mappers — dedicated handlers for Linear/Conv/LayerNorm/MultiheadAttention, structural shortcuts (`ReshapeMapper`/`PermuteMapper`/`ConcatMapper`) for view/reshape/flatten/permute/transpose/cat, generic ComputeOp fallback for everything else |
| `mapper_graph_fx.py` | `MapperGraphFxMixin`: `_partition_fx_args` splits FX args into mapper sources / bound tensors / extra args / kwargs; `_emit_generic_compute_op` builds a `ComputeOpMapper` over a `ComputeAdapter`; bound tensors are squeezed of a leading singleton dim |
| `representability_analyzer.py` | `RepresentabilityAnalyzer` classifies every FX node (grouped convs are the unsupported case) and builds the BN/activation absorption plan; `RepresentabilityReport`, `OpInfo`, `RepresentabilityError` |
| `torch_graph_tracer.py` | `trace_model`: FX symbolic tracing (MultiheadAttention/RNN/Transformer layers as leaves) + ShapeProp annotation; lock-serialized with stale-FX-patch restoration; raises `TracingError` |

## Dependencies
- **mapping** — the target IR: Mapper classes from `mapping.mapping_utils` and
  `mapping.mappers.*` (`ComputeOpMapper`, `InputMapper`, `PerceptronMapper`,
  `Conv1DPerceptronMapper`, `Conv2DPerceptronMapper`, `ConcatMapper`,
  `ReshapeMapper`, `PermuteMapper`, `Ensure2DMapper`), `ModelRepresentation`
  as the converted graph container, and `mapping.support.compute_modules`
  (`ComputeAdapter`, `_cat_along`) for generic host compute ops.
- **models** — `models.perceptron_mixer.perceptron` (`Perceptron`, the packaged
  MM+BN+activation unit) and `models.perceptron_mixer.perceptron_flow`
  (`PerceptronFlow`, base class of `ConvertedModelFlow`).

## Dependents
- **pipelining** — `pipeline_steps/config/torch_mapping_step.py` calls
  `convert_torch_model` and handles `ConversionProbeError`.
- **models** — `pretrained_bridge.py` converts pretrained models via
  `convert_torch_model` + `mark_encoding_layers`.
- **mapping** — `verification/wizard_layout_verify.py` and
  `verification/onchip_fraction.py` convert models for layout verification
  and on-chip-fraction analysis; `support/bias_compensation.py` consults
  `encoder_deploys_as_staircase_hop` for the placement-aware sync entry
  half-step fold.
- **search** — `problems/joint/layout_hook.py` converts candidate models
  during joint architecture search.
- **tuning** — `tuners/ttfs_cycle_adaptation_tuner.py` uses
  `segment_entry_perceptrons` to place its q(x) STE at segment seams.

## Exported API
- `convert_torch_model` — convert a trained native model to a `ConvertedModelFlow`.
- `check_representability` — trace + normalize + classify without converting.
- `RepresentabilityReport` — the per-op supported/unsupported classification result.

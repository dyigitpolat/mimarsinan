# torch_mapping/ -- PyTorch Model Conversion

Converts native PyTorch `nn.Module` models into mimarsinan's Mapper DAG /
Supermodel representation. The conversion happens as a pipeline step after
pretraining so that native models train at full speed and only get wrapped
in Perceptrons when the adaptation/quantization stages need them.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `torch_graph_tracer.py` | `trace_model` | FX-based graph extraction with shape propagation |
| `representability_analyzer.py` | `RepresentabilityAnalyzer`, `RepresentabilityReport`, `OpInfo` | Validates whether an FX graph can be represented in mimarsinan IR |
| `mapper_graph_converter.py` | `MapperGraphConverter` | Converts FX graph to Mapper DAG. Linear / Conv / LayerNorm / MultiheadAttention have dedicated handlers (they package weights / multi-input semantics). **All other ops** — `+`, `getitem`, `mean`, `F.*`, custom `nn.Module`s, … — flow through one generic `_emit_generic_compute_op(node, fn)` path that uses `_partition_fx_args` to classify every FX arg into `sources` (mapper-backed Nodes), `bound_tensors` (`get_attr`-backed Tensors / Parameters), `extra_args` (ints, slices, tuples, ...) and `kwargs` (non-Node kwargs), then builds a `ComputeOpMapper(sources, ComputeAdapter(fn, ...))`. **Structural shortcuts** (`torch.cat`, `torch.flatten`, `.view` / `.reshape` / `.flatten` methods, `.permute` / `.transpose` methods) are the only per-op handlers; they emit shape-only mappers (`ConcatMapper`, `ReshapeMapper`, `PermuteMapper`) that fold at mapping time without a runtime ComputeOp. |
| `converter_handlers/` | `LinearConvertMixin`, `ConvConvertMixin`, `StructuralConvertMixin` | Linear and Conv mixins package perceptron candidates with absorbed BN / activation. `StructuralConvertMixin` holds only the structural shortcuts (`_convert_cat`, `_convert_flatten_func`, `_convert_flatten_module`) and BN/activation absorption helpers — no compute-op-specific handlers. |
| `converted_model_flow.py` | `ConvertedModelFlow` | PerceptronFlow subclass wrapping the converted ModelRepresentation; overrides `_apply` so that `to(device)` / `to(dtype)` also propagates to every node in the mapper graph, keeping mapper submodules (e.g. LayerNorm, ModuleMapper.module) on the same device/dtype as the rest of the model |
| `graph_normalization.py` | `normalize_fx_graph` | MM+ fusion: folds BN into preceding Linear, fuses consecutive Linears (through Identity/BN), dead code elimination |
| `fx_shape_utils.py` | `node_input_shapes`, `node_output_shape`, `strip_batch` | Shared FX `tensor_meta` extraction. Multi-input ops record one batch-stripped shape per source so downstream IR mapping, scale analysis, and the runtime broadcast guard all see the full picture. Replaces the older `_get_input_shape` (args[0]-only) duplicates. |
| `mapper_graph_fx.py` | `MapperGraphFxMixin`, `_normalize_bound_tensor` | Generic compute-op emission + bound-tensor normalisation.  Bound tensors stored via `_partition_fx_args` are squeezed of any leading singleton dim so `ComputeAdapter`'s `unsqueeze(0).expand(B, *t.shape)` re-adds the batch dim correctly.  Parameters like ViT `pos_embedding` `(1, N, D)` and `(1, D)` biases reach the adapter as `(N, D)` and `(D,)` respectively. |
| `conversion_probe.py` | `probe_forward`, `ProbeResult`, `ConversionProbeError` | Strict-by-default warmup forward.  Parses the `"[ModelRepresentation] forward failed at node ..."` cause-chain marker so failure messages name the offending mapper.  Replaces the old `try / print / continue` patterns in `convert_torch_model` and `TorchMappingStep._verify_equivalence`. |
| `converter.py` | `convert_torch_model`, `check_representability` | Public API facade.  `convert_torch_model(..., strict=True)` is the default — warmup failures now propagate as `ConversionProbeError` with the offending node name.  Pass `strict=False` only for tests / configs that explicitly accept a known-broken forward.  `encoding_layer_placement="subsume"` (default) / `"offload"` selects the encoding-layer deployment (see `encoding_layers.py`). |
| `encoding_layers.py` | `mark_encoding_layers` | Subsume-vs-offload switch for encoding-layer neuralOps. `placement="subsume"` (default) marks segment-start perceptrons so the mappers emit them as **host ComputeOps** (spike-train generators). `placement="offload"` clears the flag so they map as on-chip **NeuralCores** and the flow encodes the raw segment input directly (Uniform/TTFS) — larger hardware-accelerated surface, functionally identical under signed-IF (offload HCM output == subsume to 1e-6). |

## Dependencies

- **Internal**: `models.perceptron_mixer.perceptron` (`Perceptron`),
  `models.perceptron_mixer.perceptron_flow` (`PerceptronFlow`),
  `models.supermodel` (`Supermodel`),
  `models.preprocessing.input_cq` (`InputCQ`),
  `mapping.mapping_utils` (all Mapper classes),
  `models.layers` (`LeakyGradReLU`).
- **External**: `torch`, `torch.fx`.

## Dependents

- `pipelining.pipeline_steps.torch_mapping_step` uses `convert_torch_model`.
- `models.builders` (torch_* builders) produce native models consumed by this module.

## Perceptron packaging rule

Conversion produces one `Perceptron` (wrapped by a `PerceptronMapper`) for each
segment of the FX graph matching the pattern **MM+ → BN? → ACT**:

- **MM+**: one or more matrix-multiplication-equivalent ops (`nn.Linear`,
  `BatchNorm` — a diagonal MM) fused into a single `nn.Linear` by
  `graph_normalization`.  Identity and BN between consecutive Linears are
  walked through; BN is folded into the preceding Linear before the pair
  is fused.
- **BN?**: optional `BatchNorm1d`/`BatchNorm2d` (absorbed into the preceding
  Linear).
- **ACT**: optional activation (`ReLU`, `LeakyReLU`, `GELU`, `Identity`),
  absorbed into the same Perceptron.

Whether the resulting Perceptron participates in pipeline processing is
determined by [`is_perceptron_activation()`](../mapping/mappers/base.py)
(True for any non-Identity activation). Any detected nonlinearity (ReLU, GELU,
LeakyReLU, etc.) maps to a NeuralCore. Identity (no activation) produces a
host-side linear ComputeOp.

The mapper boundary is the **single source of truth** for activation eligibility:
`owned_perceptron_groups()` on every mapper type (FC, Conv2D, Conv1D) returns `[]`
for Identity perceptrons and the perceptron list otherwise.  Downstream pipeline
steps such as `ActivationAdaptationStep` consume `model.get_perceptrons()` and
are therefore guaranteed to see only perceptrons with nonlinear activations — no
special-casing of `Identity` is needed outside the mapper layer.

## Exported API (\_\_init\_\_.py)

`convert_torch_model`, `check_representability`, `RepresentabilityReport`.

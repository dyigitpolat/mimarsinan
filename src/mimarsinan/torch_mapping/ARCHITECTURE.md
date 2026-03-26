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
| `mapper_graph_converter.py` | `MapperGraphConverter` | Converts FX graph to Mapper DAG; uses converter mixins for linear / conv / structural ops and handles traced transformer patterns inline (`LayerNorm`, class-token prepend, positional add, CLS select, `MultiheadAttention`) |
| `converter_handlers/` | `LinearConvertMixin`, `ConvConvertMixin`, `StructuralConvertMixin` | Per-category `_convert_*` methods mixed into `MapperGraphConverter`; transformer-specific traced patterns are handled directly in `mapper_graph_converter.py` |
| `converted_model_flow.py` | `ConvertedModelFlow` | PerceptronFlow subclass wrapping the converted ModelRepresentation; overrides `_apply` so that `to(device)` / `to(dtype)` also propagates to every node in the mapper graph, keeping mapper submodules (e.g. LayerNorm, ModuleMapper.module) on the same device/dtype as the rest of the model |
| `graph_normalization.py` | `normalize_fx_graph` | MM+ fusion: folds BN into preceding Linear, fuses consecutive Linears (through Identity/BN), dead code elimination |
| `converter.py` | `convert_torch_model`, `check_representability` | Public API facade |

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

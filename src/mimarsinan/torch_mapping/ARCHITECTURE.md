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
| `mapper_graph_converter.py` | `MapperGraphConverter` | Converts FX graph to Mapper DAG with Perceptron wrappers and weight transfer |
| `converted_model_flow.py` | `ConvertedModelFlow` | PerceptronFlow subclass wrapping the converted ModelRepresentation; overrides `_apply` so that `to(device)` / `to(dtype)` also propagates to every node in the mapper graph, keeping mapper submodules (e.g. LayerNorm, ModuleMapper.module) on the same device/dtype as the rest of the model |
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

## Exported API (\_\_init\_\_.py)

`convert_torch_model`, `check_representability`, `RepresentabilityReport`.

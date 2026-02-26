# models/ -- Neural Network Models

Contains model definitions, activation/decorator layers, spiking simulators,
and architecture-specific implementations.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `layers.py` | `LeakyGradReLU`, `TransformedActivation`, `ClampDecorator`, `QuantizeDecorator`, `SavedTensorDecorator`, `FrozenStatsNormalization`, `MaxValueScaler`, ... | Activation functions, composable decorators, normalization wrappers |
| `supermodel.py` | `Supermodel` | Top-level model wrapper (preprocessor + input activation + perceptron flow) |
| `unified_core_flow.py` | `SpikingUnifiedCoreFlow`, `StableSpikingUnifiedCoreFlow` | PyTorch-based spiking simulator for `IRGraph` |
| `hybrid_core_flow.py` | `SpikingHybridCoreFlow` | PyTorch-based spiking simulator for `HybridHardCoreMapping` |
| `simple_conv.py` | `SimpleConvMapper` | Simple convolutional model using mapper graph |
| `vgg16.py` | `VGG16Mapper` | VGG-16 architecture using mapper graph |
| `mlp_mixer_ref.py` | `MLPMixer` | Reference MLP-Mixer (not used in pipeline) |

### Subdirectories

| Directory | Purpose |
|-----------|---------|
| `perceptron_mixer/` | Core building blocks: `Perceptron`, `PerceptronFlow`, architecture implementations |
| `preprocessing/` | Input preprocessing modules (`InputCQ`) |
| `builders/` | Model builder classes for pipeline/search integration |

## Dependencies

- **Internal**: `mapping` (for `Mapper` classes, IR types, chip latency), `code_generation` (via mapping).
- **External**: `torch`, `numpy`, `einops`.

## Dependents

- `mapping` imports `Perceptron` for weight extraction during IR mapping
- `tuning` imports layer decorators for activation management
- `model_training` imports `SavedTensorDecorator`, `Perceptron`
- `pipelining` imports models, layers, and spiking simulators
- `gui` imports IR types transitively via snapshot extraction

## Exported API (\_\_init\_\_.py)

Layer types from `layers.py` and `Supermodel`. Spiking simulators are imported
directly from their files (`models.unified_core_flow`, `models.hybrid_core_flow`)
to avoid heavy import-time costs.

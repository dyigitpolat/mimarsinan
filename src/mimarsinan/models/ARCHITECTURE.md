# models/ -- Neural Network Models

Contains model definitions, activation/decorator layers, spiking simulators,
and architecture-specific implementations.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `activations.py` | `LeakyGradReLUFunction`, `LeakyGradReLU`, `StaircaseFunction`, `DifferentiableClamp` | Custom autograd activations |
| `decorators.py` | `SavedTensorDecorator`, `StatsDecorator`, `ShiftDecorator`, `ClampDecorator`, `LearnableClampDecorator`, `QuantizeDecorator`, `ActivationReplacementDecorator`, `RateAdjustedDecorator`, `MixAdjustmentStrategy`, `NestedDecoration`, `DecoratedActivation` | Composable decorators and adjustment strategies (stochastic-mixing variants -- `NoisyDropout`, `NoiseDecorator`, `ScaleDecorator`, `RandomMaskAdjustmentStrategy`, `NestedAdjustmentStrategy`, `AnyDecorator` -- removed in Phase D3; the annealed soft-quantisation `DSQQuantizer` / `DSQQuantizeDecorator` pair was removed in the post-Phase-D cleanup -- at low β it collapsed activations to ~0.5, which produced catastrophic tuner cycles; activation quantisation now uses the hard `QuantizeDecorator` (`StaircaseFunction`) only) |
| `layers.py` | Re-exports from `activations` and `decorators`; `TransformedActivation`, `FrozenStatsNormalization`, `MaxValueScaler`, `FrozenStatsMaxValueScaler` | Thin re-exporter plus standalone layers |
| `supermodel.py` | `Supermodel` | Top-level model wrapper. Forward applies preprocessor then perceptron flow only (input activation is applied once inside preprocessor, e.g. InputCQ). |
| `unified_core_flow.py` | `_ttfs_activation_from_type`, `SpikingUnifiedCoreFlow`, `StableSpikingUnifiedCoreFlow` | Helper `_ttfs_activation_from_type` resolves IR `activation_type` (including compound strings like "LeakyReLU + ClampDecorator, QuantizeDecorator") to `F.relu`/`F.leaky_relu`/`F.gelu`. Spiking simulator supports `NeuralCore.hardware_bias` in all forward paths (rate-coded, TTFS continuous, TTFS quantized). TTFS ComputeOp execution rescales inputs from [0,1] back to training range via `_ttfs_node_output_scale` so module bias terms remain correct. |
| `hybrid_core_flow.py` | `SpikingHybridCoreFlow` | PyTorch-based spiking simulator for `HybridHardCoreMapping`. Supports `HardCore.hardware_bias` in both rate-coded (bias added every cycle) and TTFS (bias added before activation/threshold) paths. TTFS ComputeOp execution rescales inputs from [0,1] back to training range via `node_activation_scales` (stored on `HybridHardCoreMapping`) so module bias terms remain correct. |
| `torch_mlp_mixer.py` | `TorchMLPMixer` | Native PyTorch MLP-Mixer (plain `nn.Module`) for torch_mapping conversion |
| `torch_mlp_mixer_core.py` | `TorchMLPMixerCore`, `_TokenMixerCore`, `_ChannelMixerCore` | MLP-Mixer variant with activation after every FC so all mixer layers package as chip perceptrons |
| `mlp_mixer_ref.py` | `MLPMixer` | Reference MLP-Mixer (not used in pipeline) |

### Subdirectories

| Directory | Purpose |
|-----------|---------|
| `perceptron_mixer/` | Core building blocks: `Perceptron`, `PerceptronFlow`, `SimpleMLP` (single mapper-repr example) |
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

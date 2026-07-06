# models/ — Torch model zoo, NN building blocks, and the deployable spiking simulator

This module owns everything that *is* a `nn.Module` in the deployment pipeline: the
trainable source architectures (MLP-Mixer variants, depth-probe MLP/CNN vehicles,
LeNet-5, SqueezeNet, pretrained torchvision bridges), the custom layers/activations/
decorators they are built from (`nn/`), and the spiking executors that run a mapped
`HybridHardCoreMapping` as a torch model (`spiking/`, centered on
`SpikingHybridCoreFlow`). Builders in `builders/` register each architecture with the
pipeline's `ModelRegistry` so configs can instantiate models by `model_type` id.

## Key files
| File | Purpose |
|---|---|
| `deep_mlp.py` | `DeepMLP`: narrow configurable-depth Linear+ReLU stack (optional equal-width residual pairs); the depth-probe vehicle. |
| `deep_cnn.py` | `DeepCNN`: configurable-depth (4..16) plain Conv-BN-ReLU stack with capped periodic MaxPool; the trainable deep-conv vehicle. |
| `lenet5.py` | `LeNet5`: classic LeNet-5 CNN, input-shape-adaptive; the classical baseline rung. |
| `squeezenet.py` | `SqueezeNet`/`FireModule`: scaled, input-adaptive Fire-module conv vehicle (opt-in; not pipeline-registered). |
| `torch_mlp_mixer.py` | `TorchMLPMixer`: native plain-`nn.Module` MLP-Mixer for `torch_mapping` conversion. |
| `torch_mlp_mixer_core.py` | Mixer variant with an activation after every FC so all mixer layers package as perceptrons. |
| `mlp_mixer_ref.py` | Third-party reference MLP-Mixer (einops-based), kept for architecture comparison; not pipeline-native. |
| `pretrained_bridge.py` | `load_pretrained_resnet18/50` (torchvision weights, resized `fc` head) + `deploy_and_eval`: run a stock model through the real convert→map→deploy SNN path and return a `DeployedEval`. |
| `builders/` | Per-architecture builder classes, the canonical `BUILDERS_REGISTRY` (`model_type` id → builder), and wizard config-schema aggregation. |
| `nn/` | NN building blocks: custom autograd activations (`LIFActivation`, TTFS nodes, STE input quantizers), composable activation decorators, standalone layers (`TransformedActivation`, `norm_affine_params`), and shared LIF/TTFS cycle kernels. |
| `perceptron_mixer/` | Perceptron-based architectures: `Perceptron`, `PerceptronFlow`, `SimpleMLP` (mapper-repr example), and the skip-connection mixer. |
| `preprocessing/` | Empty placeholder package (legacy `InputCQ` removed; input encoding now lives in IR encoding layers). |
| `spiking/` | The deployable spiking simulator: `SpikingHybridCoreFlow` (`hybrid/` stage-IO/LIF/rate/TTFS mixins), per-cycle neuron policies, TTFS wire-semantics kernel pairs (torch+numpy twins), spiking config constants, and differentiable spike-train training forwards. |

## Dependencies
- **`mapping`** — `HybridHardCoreMapping`/`HybridStage` and `IRSource` consumed by the
  spiking flow; core geometry and spike-source spans for stage IO; IR/chip latency for
  identity flows and gating; bias compensation; `mapping_utils`/`ComputeAdapter` for the
  perceptron-mixer models.
- **`chip_simulation`** — hybrid stage runner/execution and spike modes; spiking-mode
  policy and TTFS semantics predicates; run/segment spike recording (`RunRecord`,
  `SegmentSpikeRecord`); the TTFS executor.
- **`spiking`** — segment-boundary transcoding SSOT (`BoundaryConfig`,
  `encode_segment_input`) and segment-forward helpers for training forwards.
- **`pipelining`** — `ModelRegistry`, into which every builder registers its model type.
- **`tuning`** — `LazyExecutorForward` for the blended-genuine and prefix-genuine training forwards.
- **`common`** — `env.cuda_debug_enabled` debug flag in decorator transforms.

## Dependents
- `chip_simulation` — runs `SpikingHybridCoreFlow` and shares spiking config/kernels.
- `mapping` — maps perceptron/torch models; uses layer and activation types.
- `torch_mapping` — converts the torch model zoo into IR.
- `transformations` — transforms perceptrons and decorated activations.
- `tuning` — installs/adapts activation decorators and spiking forwards.
- `model_training` — trains the model zoo and spiking training forwards.
- `pipelining` — builds models via `BUILDERS_REGISTRY` in pipeline steps.
- `spiking` — boundary/encoding helpers typed against model spiking classes.
- `gui` — wizard schema and model snapshots.

## Exported API
`__init__.py` re-exports:
- Layer/activation/decorator types from `nn/layers.py`: `LeakyGradReLU`,
  `DifferentiableClamp`, `StaircaseFunction`, `NoisyDropout`, `TransformedActivation`,
  `DecoratedActivation`, `ClampDecorator`, `QuantizeDecorator`, `ShiftDecorator`,
  `ScaleDecorator`, `SavedTensorDecorator`, `StatsDecorator`, `RateAdjustedDecorator`,
  `FrozenStatsNormalization`, `MaxValueScaler`, `FrozenStatsMaxValueScaler`.
- `SqueezeNet`, `FireModule` — opt-in conv vehicle.
- `load_pretrained_resnet18`, `load_pretrained_resnet50` — pretrained bridge loaders.

The spiking simulator is intentionally not re-exported (heavy import); import
`SpikingHybridCoreFlow` from `mimarsinan.models.spiking.hybrid` directly.

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
| `vehicles/` | Deployment-native vehicle models: `stream_cnn.py` (StreamCNN — spiking-native stride-2 conv stack, no pooling, activated FC block + bare-Linear readout; streamable by construction, the P4 streamed-lif conv vehicle) |
| `squeezenet.py` | `SqueezeNet`/`FireModule`: scaled, input-adaptive Fire-module conv vehicle (opt-in; not pipeline-registered). |
| `torch_mlp_mixer.py` | `TorchMLPMixer`: native plain-`nn.Module` MLP-Mixer for `torch_mapping` conversion. |
| `vit_leaf.py` | `LeafVisionTransformer`/`LeafViTBlock`: MHA-leaf vision-transformer family (pre-LN blocks, `nn.MultiheadAttention(batch_first=True)` as the hosted FX leaf, GELU MLP, conv patchify, cls/pos embeddings) — the declared ViT conversion subset; `tiny_test_vit`/`deit_tiny_leaf`/`cifar_vit_leaf` (32px/patch 4, d=192, 3 heads, depth 7, mlp x2) configs plus the timm weight-porting shim (`timm_vit_to_leaf_state_dict`, `load_timm_vit_state_dict`: fused qkv → `in_proj_*`, fail-loud on unmapped keys). |
| `cifar_models.py` | The 32px BC-2 checkpoint vehicles: `CifarResNet`/`CifarBasicBlock` (standard CIFAR ResNet family, He et al. sec 4.2 — 3 stages x n BasicBlocks, 16/32/64, projection shortcuts, avgpool head; `cifar_resnet20()`; the residual `+` rides as a host ComputeOp) and `CifarVGG8` (6 conv+BN+ReLU in 3 maxpool stages, avgpool(2), 2 FC; `cifar_vgg8()`; worst fan-in 2304 + bias row fits a 2560-axon core without input splitting). |
| `families/` | Second-round CIFAR families: `densenet_bc.py`, 32px DenseNet-BC (`CifarDenseNetBC`: 2k stem, three dense blocks of (L-4)/6 bottleneck layers 1x1 to 4k then 3x3 to k, compression transitions with 2x2 average pool, GAP, FC; `published` pre-activation order or `post_activation` Conv-BN-ReLU order where every BatchNorm follows its conv) and `pyramidnet.py`, additive PyramidNet (`CifarPyramidNet`: basic BN-Conv-BN-ReLU-Conv-BN units widening by alpha, zero-padded identity shortcuts, 2x2 average-pool shortcut at downsampling); builders `cifar_densenet_bc`, `cifar_pyramidnet` (in `builders/torch/families/`). The published orders put BatchNorms on concatenations and residual sums, which conversion does not yet absorb (audit D1). G8 (E294) adds `cifar_resnet.py`, the CIFAR ResNet family over declared options (`CifarResNetFamily`: 3x3 stem of `stem_width` (default the base width), three stages of n blocks at base x k x (1, 2, 4), `block_kind` basic or bottleneck (expansion 4), `block_order` post-activation (conv-BN-ReLU, ReLU after the sum) or pre-activation (BN-ReLU-conv, final BN-ReLU), `shortcut` projection (1x1 conv, with BN post-activation, on the pre-activated input pre-activation) or zero-padded identity (stride subsampling, zero channels), n up to 111); its defaults and every `cifar_resnet20` configuration are `CifarResNet` module for module under the same seed; members by configuration: ResNet-110, WRN-16-8 (pre or post), pre-activation ResNet-164 and ResNet-1001; an undeclared option is refused by name (`ResNetFamilyOptionError`, a `ToolchainRefusal` and `ValueError`); builder `cifar_resnet_family` (in `builders/torch/families/`). |
| `torch_mlp_mixer_core.py` | Mixer variant with an activation after every FC so all mixer layers package as perceptrons. |
| `mlp_mixer_ref.py` | Third-party reference MLP-Mixer (einops-based), kept for architecture comparison; not pipeline-native. |
| `pretrained_bridge.py` | `load_pretrained_resnet18/50` (torchvision weights, resized `fc` head) + `deploy_and_eval`: run a stock model through the real convert→map→deploy SNN path and return a `DeployedEval`. |
| `builders/` | Per-architecture builder classes and wizard config-schema aggregation; `BUILDERS_REGISTRY` is the `ModelRegistry.builder_classes()` view (one SSOT populated by the `@ModelRegistry.register` decorators). Builders may declare a `workload_profile()` classmethod returning a `ModelWorkloadProfile`. `build_model(builder, model_config, encoding_placement=..., packaging=...)` is THE way the framework turns a builder into a model: builders BUILD and never decide `encoding_layer_placement` (a baked decision made the knob a silent no-op for the one `native` model, whose flow never takes the FX path where placement used to be applied), so a builder that returns a mapper flow has its placement resolved here at flow birth, and a builder that returns a plain torch module resolves it later in `convert_torch_model`. Both go through the single writer `mark_encoding_layers`, which the deployment's packaging contract also reaches, so a value-domain build is stamped not-applicable rather than left unstamped; `tests/unit/architecture/test_encoding_placement_ratchet.py` holds builders to writing nothing. |
| `nn/` | NN building blocks: custom autograd activations (`LIFActivation`, TTFS nodes, STE input quantizers), composable activation decorators, standalone layers (`TransformedActivation`, `norm_affine_params`), and shared LIF/TTFS cycle kernels. `activations/integer_law.py` is the integer readout law's single module (E282.L), read by the trained forward, the value executor, the int64 golden and the solver: `round_half_away` (ties away from zero, bit-equal to `torch.round` off ties) and `rnd_shift` / `rnd_shift_int64` (the hardware shift form `(z + 2^(s-1) - [z<0]) >> s`), the law `y = clamp(rnd_s(M0 f_adc(acc) + sum_k M_k q_k + B), lo, hi)` in torch (fp64 on integers) and int64 forms, f_adc (`2^t clamp(rnd_t(acc), -L, L)`, the identity at or above the accumulator width A; `accumulator_bounds` gives each column's exact bound `sum_i |W_ij| L_in` on the integer codes, the q_min code `-2^(b-1)` included, and `accumulator_width` its signed width; `saturation_reach` gives the largest |acc| whose law output is not yet saturated, from the exact thresholds of the half-away shift, which sizes a narrow ADC's step), sum and max instance reductions, BatchNorm after ReLU or ReLU6 as signed clamps, `grid_codes` (integer codes of grid values, `OffGridInputError` by name off the grid) and the 2^53 fp64 bound (`IntegerLawRangeError`); the value entry snap `value_quantizer.quantize_to_value_grid` rounds by the same `round_half_away`, with or without a readout law. `activations/readout_quantizer.py` holds `ReadoutQuantizer`, which the Readout Law step puts in every perceptron's scaler slot (no tuner rebuilds that slot): an STE onto the output grid until Soft Core Mapping stamps the solved law, then the exact law on the integer accumulator the owning mapper's own forward computes from the input codes (captured by a `ReadoutInputTap` on the input wire) and the integer weight grid, never from the fp32 perceptron output. |
| `perceptron_mixer/` | Perceptron-based architectures: `Perceptron`, `PerceptronFlow`, `SimpleMLP` (mapper-repr example), and the skip-connection mixer. |
| `preprocessing/` | Empty placeholder package (legacy `InputCQ` removed; input encoding now lives in IR encoding layers). |
| `spiking/` | The deployable spiking simulator: `SpikingHybridCoreFlow` (`hybrid/` stage-IO/LIF/rate/TTFS mixins plus the [C2] `membrane_readout` charge decode for final-only LIF output cores, applied only at the decode-to-logits boundary — spike-count records stay raw; `hybrid/carry.py` is the RASTER CARRY seam — under streamed semantics a pass boundary INSIDE a segment is a physical cut, not a semantic one, so it must hand the next pass the spike raster rather than a count, and the packed executor records that raster in producer-local time from the same `fires` the counts accumulate (`raster.sum(0) == counts` by construction). Only wires read by a later PASS of the same segment are published: a wire crossing to a later SEGMENT is a host boundary and collapses by design. An execution path that cannot record a raster — synchronized, recording, single-spike — REFUSES by name rather than handing the next pass a re-encoded count), per-cycle neuron policies, TTFS wire-semantics kernel pairs (torch+numpy twins), spiking config constants, and differentiable spike-train training forwards. |

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
- **`pipelining`** — `ModelRegistry`, into which every builder registers its model type; `BUILDERS_REGISTRY` is its `builder_classes()` view.
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

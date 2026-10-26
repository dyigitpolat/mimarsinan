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
| `nn/` | NN building blocks: custom autograd activations (`LIFActivation`, TTFS nodes, STE input quantizers), composable activation decorators, standalone layers (`TransformedActivation`, `norm_affine_params`), and shared LIF/TTFS cycle kernels. `off_floor.py` is the constant-OFF floor registry (D19): `declare_off_floor(cls, predicate)` declares that a node class outputs 0 for every pre-activation at or below 0 (a spiking node: never fires under per-cycle inputs at or below 0), `has_off_floor(node)` reads the nearest declared class in the node's MRO, and an undeclared node (`nn.Identity`, GELU, LeakyReLU) has none. `activations/integer_law.py` is the integer readout law's single module (E282.L), read by the trained forward, the value executor, the int64 golden and the solver: `round_half_away` (ties away from zero, bit-equal to `torch.round` off ties) and `rnd_shift` / `rnd_shift_int64` (the hardware shift form `(z + 2^(s-1) - [z<0]) >> s`), the law `y = clamp(rnd_s(M0 f_adc(acc) + sum_k M_k q_k + B), lo, hi)` in torch (fp64 on integers) and int64 forms, f_adc (`2^t clamp(rnd_t(acc), -L, L)`, the identity at or above the accumulator width A; `accumulator_bounds` gives each column's exact bound `sum_i |W_ij| L_in` on the integer codes, the q_min code `-2^(b-1)` included, and `accumulator_width` its signed width; `saturation_reach` gives the largest |acc| whose law output is not yet saturated, from the exact thresholds of the half-away shift, which sizes a narrow ADC's step), sum and max instance reductions, BatchNorm after ReLU or ReLU6 as signed clamps, `grid_codes` (integer codes of grid values, `OffGridInputError` by name off the grid) and the 2^53 fp64 bound (`IntegerLawRangeError`); the value entry snap `value_quantizer.quantize_to_value_grid` rounds by the same `round_half_away`, with or without a readout law. `activations/readout_quantizer.py` holds `ReadoutQuantizer`, which the Readout Law step puts in every perceptron's scaler slot (no tuner rebuilds that slot): an STE onto the output grid until Soft Core Mapping stamps the solved law, then the exact law on the integer accumulator the owning mapper's own forward computes from the input codes (captured by a `ReadoutInputTap` on the input wire) and the integer weight grid, never from the fp32 perceptron output. [N2] `ReadoutQuantizer.set_groups`: consumer groups, each group of output channels its own range and width (absent on a per-perceptron quantizer, whose state is unchanged). |
| `perceptron_mixer/` | Perceptron-based architectures: `Perceptron`, `PerceptronFlow`, `SimpleMLP` (mapper-repr example), and the skip-connection mixer. `perceptron.declare_value_domain_input` / `reads_value_domain` are the value-domain reader stamp that `torch_mapping.encoding_layers.mark_encoding_layers` writes on every perceptron of a value-domain flow at birth (its inputs are signed values, not the event wire's spike rates). |
| `preprocessing/` | Empty placeholder package (legacy `InputCQ` removed; input encoding now lives in IR encoding layers). |
| `spiking/` | The deployable spiking simulator: `SpikingHybridCoreFlow` (`hybrid/` stage-IO/LIF/rate/TTFS mixins plus the [C2] `membrane_readout` charge decode for final-only LIF output cores, applied only at the decode-to-logits boundary — spike-count records stay raw; `hybrid/carry.py` is the RASTER CARRY seam — under streamed semantics a pass boundary INSIDE a segment is a physical cut, not a semantic one, so it must hand the next pass the spike raster rather than a count, and the packed executor records that raster in producer-local time from the same `fires` the counts accumulate (`raster.sum(0) == counts` by construction). Only wires read by a later PASS of the same segment are published: a wire crossing to a later SEGMENT is a host boundary and collapses by design. An execution path that cannot record a raster — synchronized, recording, single-spike — REFUSES by name rather than handing the next pass a re-encoded count; [W33] the same module's guard, `require_backend_replay`, reads the stage-input timing rule's per-backend declaration, `chip_simulation/stage_input_clock`: the HCM, SANA-FE and Lava read a stage-input row in the reading core's local time in mimarsinan's own code, nevresim only where the nevresim tree it compiles against carries W33's read (a probe of the tree's read sites and its constant-evaluated self-test, asked only for a program with a late reader or a delayed slice); on any other backend, and on nevresim over a tree without the read, any program with a stage-input reader above latency 0, carried or host-encoded, or a delayed slice, in the units it executes, is refused by name at plan time and before nevresim compiles, on its hybrid and flat routes), per-cycle neuron policies, TTFS wire-semantics kernel pairs (torch+numpy twins), spiking config constants, and differentiable spike-train training forwards. [W15] The reads the static check judges come from `mapping/packing/executed_reads`: `hybrid/carry.replay_carried_slices` delivers each slice by `slice_deliveries`, `hybrid/segment_cache.build_segment_tensors` takes the bank fast path by `bank_read`, and `hybrid/stage_io._build_consumer_counts` memoizes `consumer_counts`. |

## Declared boundary grids (Wave 2 WI-5a)

`nn/activations/value_quantizer.py`: `BoundaryGrid.snap` (the entry snap, `quantize_to_value_grid` byte for byte),
`entry_codes` (the DAC codes the golden reads) and `entry_offset()` (0); `DeclaredBoundaryGrid` adds the offset
(a value is `code * step - offset`), the entry rounding (`ENTRY_ROUNDINGS`: nearest half away, or floor) and the
code range inside the DAC's `±(2^(b-1) - 1)` (`BoundaryGridDeclarationError` otherwise); `ValueGridQuantizer.declare`
installs one (a later `calibrate` drops it). `integer_law.grid_codes(..., offset=)` reads `(x + o) / Δ` within the
dtype's rounding. `readout_quantizer.ReadoutStamp.evaluate` reads its entry grid's offset and subtracts
`OFFSET_MULTIPLIER * (accumulator of ones)` from the bias, the law's own per-position offset term.

## Lanes in the model (Wave 2 WI-5e)

`nn/activations/lane_quantization.py` holds the model side of a lane program: the unsigned lane grid
(`quantize_to_lane_grid`, codes `0..2^b - 1` at step `range / (2^b - 1)`, an optional offset), `LaneAddModule` (a
fused add's end: STE onto its accumulator perceptron's lane grid, or once stamped that perceptron's law with the
add's other input as operand, one rounding for the sum), `LanePoolModule` and its `PoolStamp` (a pool lane: STE onto
its own grid, or once stamped the sum of codes over each window and the law), `LaneReduceModule` (a producer-side
reduction's end, D11: an average pool its perceptron alone feeds, STE of the pooled value onto that perceptron's lane
grid, or once stamped the perceptron's accumulators summed over each window and its law per pool output, one
rounding) and `LaneSeam` (the calibration seam of a lane that rounds downstream). `ReadoutQuantizer.set_lane` makes a
perceptron's readout quantizer its lane (unsigned or offset grid; `lane_continues` when it rounds at a fused add's
or a reduction's end, its own slot then passing the value through); class defaults keep every other quantizer's
pickle unchanged. `ReadoutStamp.evaluate` reads a lane's input offset, its operand (on the operand's own grid,
`OPERAND_M`, `OPERAND_STEP`, `OPERAND_OFFSET`), its output offset (`OUTPUT_OFFSET`) and a reduction's window
(`lane_reduce`, its law arrays then per pool output element as the IR folded them); without them it is step 1's law.
`DeclaredBoundaryGrid.unsigned_dac` lets a lane chip's DAC read the 2^b unsigned codes at a declared entry, and an
unsigned declared code range there is the unsigned lane grid of its bits (`DeclaredBoundaryGrid.unsigned`, `step =
scale / (2^b - 1)`, the step a lane's readout range of the same value writes).

## Lossless count carry under cascaded TTFS (Wave 2 pack)

From the pack unit's fix round (fragment `11_wave2/pack/arch_fragments/models.md`). `spiking/hybrid/flow.py`: beside
`refuse_stochastic_replay`, the constructor runs `refuse_lossy_count_carry`. A cascaded TTFS run carries a count across
a pass inside a segment through its single-spike re-encode under the collapse transfer, the time code's lossless codec
under the family's one legal input encoding (`spike_generation_mode` TTFS); under any other encoding (not a legal
declaration, reachable only by constructing the hybrid executor directly) a program that carries across a pass is
refused by name (`PassReplayRefusal`), and a program that carries nothing runs.

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

## Constant-OFF floor and declared input range (D19)

`nn/off_floor.py` declares `nn.ReLU` and `LeakyGradReLU`, `LIFActivation` while its window-start membrane (snapped
up half a quantum when a membrane lattice is armed) stays below the threshold, `TTFSActivation` and
`TTFSCycleActivation`; `tuning.orchestration.blend_ramp` declares `BlendActivation` (and so `LIFBlendActivation`)
when every side its rate reads has the floor. The constant-OFF bias clip reads the perceptron's `base_activation`
through `has_off_floor`; the ReLU adaptation keeps its own name list (`RELU_COMPATIBLE_TYPES`), since a LIF node
has the floor but is not ReLU-compatible.

`nn/activations/value_quantizer.py`: `BoundaryGrid.value_range` is `(code_min * step - entry_offset(), code_max *
step - entry_offset())` over `code_range` (the symmetric grid: `(-levels, levels)`, offset 0), so a grid that declares
an offset or a code range gives its own range through the same three members; `wire_output_grid` is the armed grid an
input wire's LAST op snaps onto (the range of what the perceptron reads; any other op at the end bounds nothing) and
`carries_value_grid` says whether a wire holds a value-domain boundary quantizer at all.

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

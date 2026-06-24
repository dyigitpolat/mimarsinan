# models/ -- Neural Network Models

Contains model definitions, activation/decorator layers, spiking simulators,
and architecture-specific implementations.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `activations.py` | `LeakyGradReLU`, `LIFActivation`, `uniform_encode_to_spike_train`, `run_cycle_accurate`, `StrictATanSurrogate`, `ChipInputQuantizer`, `TTFSInputGridQuantizer`, … | Custom autograd activations. **Encoding-input quantizers** (`nn/activations/autograd.py`): `ChipInputQuantizer` = STE `round(x·T)/T` (LIF chip rate grid); `TTFSInputGridQuantizer` = STE TTFS encode→decode round trip `(S − round(S·(1−x)))/S` (`TTFSGridSnapFunction`), bit-identical to `ttfs_encoding.ttfs_input_grid_quantize` — they differ at half-step ties. **`LIFActivation`**: SpikingJelly IFNode (subtractive reset), **signed integrate-and-fire** — charges the membrane with `x/scale` (no relu), so it can go negative and recover, matching the chip/HCM (`memb += W@s + b`); rate mode = multi-step mean; **`set_cycle_accurate(True)`** = single-step per call; **`forward_spiking`** + **`use_cycle_accurate_trains`** delegate to **`mimarsinan.spiking.spike_trains.lif_spike_train`**. **`thresholding_mode`**: `"<"` strict vs `"<="` inclusive. |
| `../spiking/spike_trains.py` | `lif_spike_train`, `uniform_spike_train`, `rates_to_spike_train` | Shared spike-train construction for PyTorch cycle-accurate path and hybrid SCM. |
| `decorators.py` | `NoisyDropout`, `SavedTensorDecorator`, `ShiftDecorator`, `RateAdjustedDecorator`, `RateBuffer`, … | Composable decorators. **Short-circuits**: omit `ShiftDecorator` when `shift_rate==0`; `RateAdjustedDecorator` / `NoisyDropout` no-op at rate 0 (the rate==0 RNG-skip of the RandomMask `torch.rand` is load-bearing for parity). `RateAdjustedDecorator.rate` may be a plain float or a `RateBuffer` (registered scalar `alpha`, `set(a)` fills in place); when a buffer, the rate is read live at transform time so an adaptation ramp advances O(1) without rebuilding the decorator stack (the W9 fix). |
| `layers.py` | Re-exports; `TransformedActivation`, `norm_affine_params`, … | Thin re-exporter + standalone layers. `norm_affine_params(norm)` = `(u, beta, mean)` of a normalization's frozen-stats affine form (differentiable; works for `BatchNorm1d/2d` and `FrozenStatsNormalization`) — SSOT consumed by `PerceptronTransformer._get_u_beta_mean` and `effective_preactivation_bias`. |
| `supermodel.py` | `Supermodel` | Top-level wrapper (preprocessor + perceptron flow) |
| `spiking/hybrid/identity_flow.py` | `build_identity_spiking_flow` | Runs a flat `IRGraph` on a 1:1 identity `HybridHardCoreMapping` (`build_identity_hybrid_mapping`, no pool/pad/reindex) through the single `SpikingHybridCoreFlow` executor. Drop-in for the retired flat-`IRGraph` spiking simulator (tests / tooling / SCM rung-2 gate); computes IR latencies only when missing. **Note:** hybrid TTFS output is count-scaled (×T), unlike the retired flow's normalized output. |
| `hybrid_core_flow.py` | `SpikingHybridCoreFlow` | **Primary** deployable simulator: `HybridHardCoreMapping`, per-core latencies, segment boundaries. Stages dispatch via `chip_simulation.hybrid_stage_runner.run_hybrid_stages` with `HybridStageContext` callbacks (`after_neural` / `after_compute`) for GPU refcount, segment weight cache, and `forward_with_recording()`. Compute stages use `resolve_stage_compute_scales` (always-on). |
| `spiking/wire_semantics.py` | `WireSemantics`, `ttfs_quantized_staircase(_np)`, `ttfs_spike_time(_np)`, `ttfs_grid_quantize(_np)`, `floor_staircase(_np)` | **Kernel pairs**: each TTFS wire op defined once with torch+numpy twins (identical op order → bit-equal in float64; cross-twin tests sweep ±1 ULP around the S-grid). `WireSemantics(S, compare_mode)` bundles the ops per deployment (`contract.wire()`); `compare_mode='<'` = nevresim `StrictCompare` tie shift (exact grid ties fire one cycle later; C++ `<` parity not yet harness-covered). |
| `ttfs_kernels.py` | `ttfs_quantized_activation` | Thin aliases over `spiking/wire_semantics.py` kernel pair (names kept for importers). |
| `lif_kernels.py` | `lif_fire_and_reset` | Shared LIF threshold + Novena/Default reset step for unified and hybrid flows. |
| `torch_mlp_mixer.py`, `torch_mlp_mixer_core.py`, `mlp_mixer_ref.py` | … | MLP-Mixer variants |
| `deep_mlp.py` | `DeepMLP` | Narrow configurable-depth `Flatten -> [Linear(width)+act] x depth -> Linear(classes)` stack (pure Linear+ReLU, no conv/attention); the depth-probe vehicle, registered as `deep_mlp` (category "torch"). **`residual=True`** (opt-in, default off) wraps consecutive pairs of equal-width hidden layers (after the stem) in a bare equal-width skip `z = z + block(z)`, which lowers to a param-free host `ComputeAdapter(operator.add)` ComputeOp (the residual-mapping path); Linear count and `hidden.*` param names unchanged so plain↔residual share a state_dict, default path byte-identical |
| `deep_cnn.py` | `DeepCNN`, `allowed_pool_count` | Configurable-depth (4..16) plain `[Conv(k3,pad1)+BatchNorm+ReLU] x depth` stack with periodic MaxPool (capped by `allowed_pool_count` so a 28x28/32x32 input never collapses below 1x1), channels doubling per pool boundary (cap 128), `AdaptiveAvgPool->Linear` head. No grouped/depthwise conv, no residuals — maps fully on-chip (SAME padding keeps multi-channel convs off the LayoutSourceView no-pad limitation). The TRAINABLE deep-conv vehicle for the cascaded single-spike firing-gain probe; registered as `deep_cnn` (category "torch"), input-shape-adaptive for MNIST (1x28x28) and SVHN (3x32x32) |

| `lenet5.py` | `LeNet5` | Classic LeNet-5 CNN (Conv 1→6 k5, Conv 6→16 k5 with `padding=2`, two MaxPool, FC 120→84→n_classes); T1 classical rung, pipeline-native (no grouped/depthwise conv) |
| `squeezenet.py` | `SqueezeNet`, `FireModule` | Scaled, input-adaptive SqueezeNet Fire-module conv vehicle (squeeze 1x1 → expand 1x1 + 3x3 with `padding=1`, `torch.cat` concat) for the SCALE breadth of the deployment hypervolume. Stem Conv(k3,pad1)+ReLU → periodic MaxPool → 6 Fire modules → Conv1x1 readout → AdaptiveAvgPool → Flatten; channels follow the squeeze/expand ratio and grow with `width` (default 24, ~379K params). Pipeline-native: Conv2d / ReLU / MaxPool2d / AdaptiveAvgPool2d / cat only — no grouped/depthwise conv, no attention/LayerNorm, maps fully on-chip (MEASURED `classify_validity` tier `VALID`, param_frac = mac_frac = 1.0 under `offload`; `estimate_cores_needed` ~942 hard cores on the default 256×256 / 1000-core budget). Input-adaptive for MNIST (1x28x28) and SVHN (3x32x32); not pipeline-registered (opt-in model) |

### Subdirectories

| Directory | Purpose |
|-----------|---------|
| `perceptron_mixer/` | `Perceptron`, `PerceptronFlow`, `SimpleMLP` |
| `preprocessing/` | `InputCQ` |
| `builders/` | Model builders for pipeline/search |

## Dependencies

- **Internal**: `mapping`, `code_generation` (via mapping), `chip_simulation.hybrid_execution`, `chip_simulation.spike_recorder`.
- **External**: `torch`, `numpy`, `einops`; lazy `spikingjelly` for LIF.

## Dependents

- `mapping`, `tuning`, `model_training`, `pipelining`, `gui` (via snapshots).

## Compute dtype (float64 default)

`SpikingHybridCoreFlow` defaults to **`torch.float64`** for on-chip spiking math. Nevresim TTFS uses `signal_t = double`; float32 can flip `ceil(S*(1 - V/θ))` when `V` is within one ULP of `θ`, which compounds across layers (~multi–percentage-point SCM drift on MNIST-scale runs). Pass `compute_dtype=torch.float32` only after measuring sensitivity; weight `nn.Parameter`s stay float32 for matmul.

**Packed buffers:** owned/bank **weights** are stored as float32 `nn.Parameter`s (matmul path). **Thresholds** and **hardware_bias** are packed in `compute_dtype` (float64 by default) so `ceil(S*(1-V/θ))` and gather/add paths match HCM/nevresim — Python float thresholds (e.g. 8.656…) do not round-trip exactly in float32.

### TTFS ComputeOp scaling (`SpikingHybridCoreFlow` / HCM)

NeuralCore TTFS outputs are normalized to `[0, 1]` via effective weights; **ComputeOps** wrap the *training* module (unscaled bias). Two scales per ComputeOp node:

| Field | Role |
|-------|------|
| `_ttfs_node_input_scale` | Multiply gathered input before `execute_on_gathered` — rescale NeuralCore sources from `[0,1]` to training range. **1.0** when all sources are raw graph input (encoding-layer ops); must not spuriously multiply encoding inputs. |
| `_ttfs_node_output_scale` | Divide module output back to `[0,1]` for downstream `W_eff`. Perceptron-wrapped ops use the wrapped `activation_scale`; generic ops use the average of source scales. |

ComputeOp gather runs in **float32** (slice assignment into `inp` downcasts; do not clone the full activation cache per op — ViT-scale graphs OOM). Results are stored in `compute_dtype` for downstream cores.

### `SpikingHybridCoreFlow` execution notes

- **Segment weight cache:** single-segment LRU — at most one neural segment's GPU weights resident (ViT-scale mappings need tens of GB if all segments are cached). `torch.cuda.empty_cache()` once per `forward`, not per segment evict.
- **State buffer:** `Dict[node_id, Tensor]` with **consumer refcount** — drop entries when downstream read count hits 0 (long compute-op chains otherwise retain GB of intermediates per forward).
- **Forward dispatch (`forward`):** routes on `SpikingModePolicy.decode_mode()` (V2) — `timing` → `_forward_ttfs` (analytical + synchronized cycle TTFS), `count` → `_forward_rate` (LIF/rate + cascaded cycle TTFS) — instead of an `is_cascaded_ttfs` + `requires_ttfs_firing` predicate cascade.
- **Per-cycle policy (`spiking/cycle_policy.py`):** the cascade executor (`_run_neural_segment_rate`) is parameterized by `cycle_neuron_policy(spiking_mode, schedule, firing_mode)` mirroring nevresim's `FirePolicy`. `LIFCyclePolicy` (`latency_gated=True`) = multi-spike reset; `TTFSGreedyCyclePolicy` (`latency_gated=False`, `single_spike_io=True`) = cascaded greedy **single-spike** fire-once with a `ramp_current` accumulator (kernels in `nn/ttfs_cycle_kernels.py::ttfs_cycle_fire_once` + `spiking/ttfs_cycle_step.py`). The executor branches on `policy.latency_gated` (gating) and `policy.single_spike_io` (single-spike I/O + ramp decode).
- **Rate-coded LIF (gated):** always-on axon contributes **one spike per cycle** in `[0, T)` only; neuron-source inputs accumulate only inside the source core's `[latency, latency+T)` window (matches Loihi `reset_offset` / HCM buffer semantics).
- **Cascaded TTFS (single-spike, latency-gated):** hardware-faithful — each neuron fires **exactly once** (one spike on the wire). Cores are **latency-gated** (active in `[lat, lat+T)`), so bias/ramp start at the core's reference time and it fires in-window. The integration is a **ramp** reconstructed at the consumer: a single arrival at `t_j` contributes its weight every later cycle (`membrane(t)=Σ w_j·(t−t_j)`), via `ramp_current`. Input is single-spike; the segment-output **value** is the ramp counted **within each source's own window** `[src_lat, src_lat+T)` (per-column arrival latch `out_arrival`), value `= (src_lat+T−fire)/T`; recorded per-neuron **count** is single-spike traffic. **Per-source windowing is essential** — full-window accumulation overcounts shallow sources by `(chip_latency−src_lat)/T` and saturates everything when latency >> T (gave 9.2% on a real deep model before the fix). Greedy & lossy but bit-consistent across HCM/nevresim/SANA-FE; schedule via `ttfs_cycle_schedule`.
- **Encoding boundary:** ComputeOps that wrap a plain LIF-Perceptron emit a `(T,B,D)` spike train via `mimarsinan.spiking.segment_encoding.emit_compute_spike_train` (single-step `T` cycles, divided by `activation_scale` for the chip's binary-input convention). Wrapper mappers (e.g. `Conv2DPerceptronMapper`), non-LIF perceptrons, and structural ops (mean/flatten/add/etc.) emit `None`; the downstream consumer uniform-encodes their rate. `build_segment_input_spike_train` consumes cached trains verbatim and raises on partial caches under cycle-accurate mode.
- **`forward_with_recording()`:** requires `batch_size==1` and a per-cycle cascade mode (`lif` or cascaded `ttfs_cycle_based`); fills `RunRecord` for Loihi/SANA-FE/nevresim parity.

### Rate vs TTFS always-on sources (`SpikingHybridCoreFlow`)

In LIF/rate mode, always-on (`kind=="on"`) sources fire every cycle. In TTFS modes, always-on fires **only at cycle 0** (bias pulse once); later cycles skip the fill so rate and TTFS paths stay consistent with hardware.

## Exported API (`__init__.py`)

Layer types from `layers.py` and `Supermodel`. The spiking simulator is imported from `hybrid_core_flow` directly (heavy import cost); flat-`IRGraph` callers use `spiking/hybrid/identity_flow.build_identity_spiking_flow`.

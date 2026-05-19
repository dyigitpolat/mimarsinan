# models/ -- Neural Network Models

Contains model definitions, activation/decorator layers, spiking simulators,
and architecture-specific implementations.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `activations.py` | `LeakyGradReLU`, `LIFActivation`, `uniform_encode_to_spike_train`, `run_cycle_accurate`, `StrictATanSurrogate`, … | Custom autograd activations. **`LIFActivation`**: SpikingJelly IFNode (subtractive reset); rate mode = multi-step mean; **`set_cycle_accurate(True)`** = single-step per call; **`forward_spiking`** returns real `(T,B,…)` trains for downstream encoding. **`thresholding_mode`**: `"<"` strict vs `"<="` inclusive. |
| `decorators.py` | `NoisyDropout`, `SavedTensorDecorator`, `ShiftDecorator`, `RateAdjustedDecorator`, … | Composable decorators. **Short-circuits**: omit `ShiftDecorator` when `shift_rate==0`; `RateAdjustedDecorator` / `NoisyDropout` no-op at rate 0. |
| `layers.py` | Re-exports; `TransformedActivation`, … | Thin re-exporter + standalone layers |
| `supermodel.py` | `Supermodel` | Top-level wrapper (preprocessor + perceptron flow) |
| `unified_core_flow.py` | `SpikingUnifiedCoreFlow` | Spiking simulator for a flat `IRGraph` (tests / tooling). Default **`compute_dtype=torch.float64`**; optional `float32`. **Weight banks**: one `nn.Parameter` per `WeightBank`. Per-forward activation/spike caches with explicit release to limit GPU memory. **Not** used for the pipeline soft-core step metric (see `hybrid_core_flow`). |
| `hybrid_core_flow.py` | `SpikingHybridCoreFlow` | **Primary** deployable simulator: `HybridHardCoreMapping`, per-core latencies, segment boundaries. Same float64 default; shared weight-bank Parameters; **`forward_with_recording()`** for Loihi/SANA-FE parity; `decref` on state-buffer consumers. Used by Soft Core Mapping (SCM metric), Hard Core Mapping verification, Loihi step, SANA-FE parity reference. |
| `torch_mlp_mixer.py`, `torch_mlp_mixer_core.py`, `mlp_mixer_ref.py` | … | MLP-Mixer variants |

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

`SpikingUnifiedCoreFlow` and `SpikingHybridCoreFlow` default to **`torch.float64`** for on-chip spiking math. Nevresim TTFS uses `signal_t = double`; float32 can flip `ceil(S*(1 - V/θ))` when `V` is within one ULP of `θ`, which compounds across layers (~multi–percentage-point SCM drift on MNIST-scale runs). Pass `compute_dtype=torch.float32` only after measuring sensitivity; weight `nn.Parameter`s stay float32 for matmul.

**Packed buffers:** owned/bank **weights** are stored as float32 `nn.Parameter`s (matmul path). **Thresholds** and **hardware_bias** are packed in `compute_dtype` (float64 by default) so `ceil(S*(1-V/θ))` and gather/add paths match HCM/nevresim — Python float thresholds (e.g. 8.656…) do not round-trip exactly in float32.

### TTFS ComputeOp scaling (`SpikingUnifiedCoreFlow` / HCM)

NeuralCore TTFS outputs are normalized to `[0, 1]` via effective weights; **ComputeOps** wrap the *training* module (unscaled bias). Two scales per ComputeOp node:

| Field | Role |
|-------|------|
| `_ttfs_node_input_scale` | Multiply gathered input before `execute_on_gathered` — rescale NeuralCore sources from `[0,1]` to training range. **1.0** when all sources are raw graph input (encoding-layer ops); must not spuriously multiply encoding inputs. |
| `_ttfs_node_output_scale` | Divide module output back to `[0,1]` for downstream `W_eff`. Perceptron-wrapped ops use the wrapped `activation_scale`; generic ops use the average of source scales. |

ComputeOp gather runs in **float32** (slice assignment into `inp` downcasts; do not clone the full activation cache per op — ViT-scale graphs OOM). Results are stored in `compute_dtype` for downstream cores.

### `SpikingHybridCoreFlow` execution notes

- **Segment weight cache:** single-segment LRU — at most one neural segment's GPU weights resident (ViT-scale mappings need tens of GB if all segments are cached). `torch.cuda.empty_cache()` once per `forward`, not per segment evict.
- **State buffer:** `Dict[node_id, Tensor]` with **consumer refcount** — drop entries when downstream read count hits 0 (long compute-op chains otherwise retain GB of intermediates per forward).
- **Rate-coded LIF:** always-on axon contributes **one spike per cycle** in `[0, T)` only; neuron-source inputs accumulate only inside the source core's `[latency, latency+T)` window (matches Loihi `reset_offset` / HCM buffer semantics).
- **Encoding layers:** ComputeOps that own a Perceptron (or conv mapper driving `forward`) can emit a real `(T,B,D)` spike train via `forward_spiking`; the next neural segment reads that train instead of re-encoding rates uniformly.
- **`forward_with_recording()`:** requires `batch_size==1` and `spiking_mode=='lif'`; fills `RunRecord` for Loihi/SANA-FE parity.

### Rate vs TTFS always-on sources (`SpikingUnifiedCoreFlow`)

In LIF/rate mode, always-on (`kind=="on"`) sources fire every cycle. In TTFS modes, always-on fires **only at cycle 0** (bias pulse once); later cycles skip the fill so rate and TTFS paths stay consistent with hardware.

## Exported API (`__init__.py`)

Layer types from `layers.py` and `Supermodel`. Spiking simulators imported from `unified_core_flow` / `hybrid_core_flow` directly (heavy import cost).

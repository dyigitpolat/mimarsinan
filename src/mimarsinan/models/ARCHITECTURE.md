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

## Exported API (`__init__.py`)

Layer types from `layers.py` and `Supermodel`. Spiking simulators imported from `unified_core_flow` / `hybrid_core_flow` directly (heavy import cost).

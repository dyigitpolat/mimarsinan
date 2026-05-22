# pipelining/pipeline_steps/ -- Individual Pipeline Step Implementations

Each file implements one `PipelineStep` subclass corresponding to a stage
in the deployment pipeline.

## Key Components

| File | Step Class | Pipeline Phase |
|------|-----------|----------------|
| `activation_utils.py` | `has_non_relu_activations`, `RELU_COMPATIBLE_TYPES` | Shared helper for activation steps |
| `architecture_search_step.py` | `ArchitectureSearchStep` | Configuration; fixed path uses `emit_model_config_entries` + `build_platform_constraints_resolved`; search results JSON via `to_json_safe`. |
| `model_configuration_step.py` | `ModelConfigurationStep` | Configuration via `emit_model_config_entries` + `build_platform_constraints_resolved`. |
| `model_building_step.py` | `ModelBuildingStep` | Model construction (`safe_warmup_forward`, `adaptation_manager_factory`) |
| `pretraining_step.py` | `PretrainingStep` | Training (`TrainerPipelineStep`, `make_basic_trainer`) |
| `activation_analysis_step.py` | `ActivationAnalysisStep` | Quantization prep (`TrainerPipelineStep`, `make_basic_trainer`; writes `activation_scales` / `activation_scale_stats`) |
| `activation_adaptation_step.py` | `ActivationAdaptationStep` | Activation adaptation (`TunerPipelineStep`) |
| `clamp_adaptation_step.py` | `ClampAdaptationStep` | Quantization/TTFS clamping (`TunerPipelineStep.run_tuner(ClampTuner, …)`) |
| `activation_shift_step.py` | `ActivationShiftStep` | Quantization (`TunerPipelineStep`; TTFS bias deferred to SCM) |
| `activation_quantization_step.py` | `ActivationQuantizationStep` | Quantization (`TunerPipelineStep.run_tuner`) |
| `weight_quantization_step.py` | `WeightQuantizationStep` | Quantization (`TunerPipelineStep.run_tuner(NormalizationAwarePerceptronQuantizationTuner)`) |
| `quantization_verification_step.py` | `QuantizationVerificationStep` | Verification (`TrainerPipelineStep`; trainer created in `process`) |
| `normalization_fusion_step.py` | `NormalizationFusionStep` | Optimization (`TrainerPipelineStep`; `fuse_into_perceptron`; `validate` uses `trainer.test()`) |
| `soft_core_mapping_step.py` | `SoftCoreMappingStep` | Mapping: `run_hcm_mapping_metric` (caches `hybrid_mapping`), `make_basic_trainer`, `run_optional_viz` for flowchart. |
| `core_quantization_verification_step.py` | `CoreQuantizationVerificationStep` | Verification (`chip_quantize.verify_ir_graph_quantized`) |
| `lif_adaptation_step.py` | `LIFAdaptationStep` | LIF adaptation (`TunerPipelineStep.run_tuner(LIFAdaptationTuner)`) |
| `noise_adaptation_step.py` | `NoiseAdaptationStep` | Optional training noise (`TunerPipelineStep`; gated by `enable_training_noise`) |
| `pruning_adaptation_step.py` | `PruningAdaptationStep` | Pruning (`TunerPipelineStep.run_tuner(PruningTuner)`) |
| `hard_core_mapping_step.py` | `HardCoreMappingStep` | Mapping: reuses cached `hybrid_mapping`; `run_hcm_mapping_metric`; optional hybrid viz via `run_optional_viz`. |
| `simulation_step.py` | `SimulationStep` | Verification |
| `loihi_simulation_step.py` | `LoihiSimulationStep` | Verification (`require_lif_spiking_mode`; parity via `simulation_factory`) |
| `sanafe_simulation_step.py` | `SanafeSimulationStep` | Verification + stats (`require_lif_spiking_mode`; `promises=["sanafe_simulation_results"]`) |
| `torch_mapping_step.py` | `TorchMappingStep` | Model conversion (`TrainerPipelineStep`, `make_basic_trainer`) |
| `weight_preloading_step.py` | `WeightPreloadingStep` | Pretrained weights (`TrainerPipelineStep`, optional `recipe`) |

## Dependencies

- **Internal**: Nearly all other modules -- this is the orchestration layer.
  Key imports: `models`, `mapping`, `transformations`, `tuning`, `model_training`,
  `data_handling`, `chip_simulation`, `visualization`, `search`.
- **External**: `torch`, `numpy`.

## Dependents

- `pipelining.pipelines.deployment_pipeline` registers all steps.

## Notes

### Per-source input scaling

Both `WeightQuantizationStep` (when enabled) and `SoftCoreMappingStep` call
`compute_per_source_scales()` to traverse the mapper graph and set
`per_input_scales` on each perceptron.  Each element of this 1-D tensor holds
the `activation_scale` of the specific source feeding that input channel.  For
sequential models all elements are identical; for layers after a `ConcatMapper`
(e.g. SqueezeNet fire modules) the values vary by channel group.  When a
dimension-rearranging mapper (e.g. `EinopsRearrangeMapper`) makes the source
channel count incompatible with the perceptron's input features, the mean of the
source scales is used as a uniform fallback.
`PerceptronTransformer` uses `per_input_scales` when computing effective weights.

### Activation adaptation: Clamp vs Activation Adaptation

`ActivationAnalysisStep` now calibrates activation scales from a bounded
multi-batch validation sample instead of a single minibatch. Bare activations
from freshly torch-mapped models are wrapped temporarily so `SavedTensorDecorator`
can observe them without mutating the long-term model structure. The scale
policy is a true count-based quantile (`torch.quantile(..., interpolation="higher")`)
over sampled positive activations, and the step emits `activation_scale_stats`
with per-layer names, sample counts, and scale summary stats for downstream
diagnostics.

**Activation Adaptation** always runs immediately after Activation Analysis in
all pipeline configurations: it replaces non-ReLU chip-targeted bases (GELU,
LeakyReLU) with ReLU when any exist; it does not apply activation_scales or
set clamp_rate.  `ActivationAdaptationStep.validate()` delegates to
`tuner.validate()`, which returns the cached metric from `_after_run()`
(measured once via `trainer.test()` after the ReLU commit).

When `activation_quantization` is True or spiking is TTFS, **Clamp
Adaptation** runs next.  `ClampAdaptationStep` **always uses `ClampTuner`**
regardless of the current activation types.

After `ActivationAdaptationStep` commits bases to `LeakyGradReLU`,
`ClampTuner` ramps `clamp_rate` from 0→1 with recovery training at each step.
Its instant probe is eval-only: candidate rates are scored with
`validate_n_batches(validation_steps)` so BatchNorm/dropout state stays intact
and the step-search metric matches the shared tuner loop.
`ClampAdaptationStep` passes through `activation_scale_stats` so `ClampTuner`
can reject scale/order mismatches before mutating model state, and the step
caches a final `trainer.test()` result so repeated `validate()` calls do not
re-sample new minibatches.

**Activation Shift** now uses `ActivationShiftTuner` instead of owning a raw
`BasicTrainer` loop. The structural shift application is unchanged: rate-coded
mode bakes the shift into effective bias and enables the shift decorator,
while TTFS still defers bias compensation until `SoftCoreMappingStep`. Only the
recovery path changed: after the shift is applied, the tuner uses the same
`TuningBudget`-derived step counts, LR search, multi-batch validation, and
cached final full-test metric style used elsewhere in the tuning stack.

Shared logic for detecting non-ReLU activations lives in `activation_utils.py`
(`has_non_relu_activations`, `needs_relu_adaptation`, `RELU_COMPATIBLE_TYPES`).

**Mapper eligibility contract**: All mapper types (`PerceptronMapper`,
`Conv2DPerceptronMapper`, `Conv1DPerceptronMapper`) implement
`owned_perceptron_groups()` using `is_perceptron_activation()` from
`mapping.mappers.base`.  This means `model.get_perceptrons()` always returns
only perceptrons with nonlinear activations — Identity (host-side ComputeOp)
perceptrons are never visible to any pipeline step.  `activation_utils.py`
therefore needs no special-casing for `Identity`; `needs_relu_adaptation` only
checks whether the activation is already ReLU-compatible.

### TTFS shift compensation

For TTFS modes with `activation_quantization`, the `QuantizeDecorator`'s nested
`ShiftDecorator` adds +shift before ReLU during training.  The effective bias
must include this shift for the IR/TTFS simulation to match the trained model.
`SoftCoreMappingStep` adds `shift / activation_scale` to the effective bias
after all training is finished but before IR mapping.

### Hardware-bias quantization scaling

When `weight_quantization=True`, `SoftCoreMappingStep` multiplies each
NeuralCore's `core_matrix` by a quantization scale `s` (≈ `q_max / max_weight`)
and sets `threshold = s`.  The TTFS simulation formula is then:

```
act(W_q @ x + b_hw) / threshold
```

For this to equal the intended `act(W_eff @ x + b_eff)`, the `hardware_bias`
must also be multiplied by `s` and rounded to the nearest integer —
otherwise the bias contribution is attenuated by a factor of `1/s`
(typically ~1/127 for 8-bit).  `SoftCoreMappingStep` applies
`np.round(hardware_bias * s)` to `node.hardware_bias` for every NeuralCore
that has one, covering both owned-weight FC layers and bank-backed Conv2D layers.

Note: the legacy always-on axon row (used when `hardware_bias=False`) is immune
to this issue because the bias row lives inside `core_matrix` and is scaled
automatically together with the weights.

### Effective axon capacity with legacy bias mode

When `hardware_bias=False`, every biased core allocates an extra always-on axon
row for bias.  `SoftCoreMappingStep` reduces the effective `max_axons` by 1
(i.e. passes `max_axons - 1` to `IRMapping`) so that wide-core detection and
layout estimation correctly account for this reserved slot.
`ModelConfigurationStep` and `ArchitectureSearchStep` propagate the `has_bias`
flag from the pipeline config to `platform_constraints_resolved` cores so that
`SoftCoreMappingStep` can resolve the hardware-bias mode correctly.

### TTFS shift idempotency (soft core mapping)

Applies only to **`ttfs_quantized`** with **`activation_quantization`** (not
continuous `ttfs`). Training uses `floor((V + shift)*tq)/tq`; IR simulation uses
`floor(V*tq)/tq` unless shift is baked into effective bias via
`_apply_ttfs_quantization_bias_compensation()` in `SoftCoreMappingStep`.

- **Encoding layers** (`is_encoding_layer`): skip — host `ComputeOp` path already
  applies `QuantizeDecorator` shift in `TransformedActivation`.
- **Idempotency:** `PerceptronTransformer.apply_effective_bias_transform` overwrites
  bias each call; mark `_ttfs_shift_baked_into_bias` after the first bake so
  resume/re-run does not double-shift.
- **Continuous `ttfs`:** do not bake shift — unclamped analytical path would push
  some outputs above 1.0; `ttfs_quantized` clamps via `k_fire.clamp(0, S-1)`.

### Deployment Accuracy Semantics

`SoftCoreMappingStep` runs the soft-core spiking simulation as the step metric
using `simulation_steps` cycles per sample plus per-core latency cycles for
warmup, and the same `max_simulation_samples` / `seed` policy used by downstream
simulation steps. Continuous `ttfs` metrics use `SpikingUnifiedCoreFlow` via
`build_spiking_flow_for_metric` (flat IR TTFS matches training); segmented
`SpikingHybridCoreFlow` for the same mode can diverge on pruned graphs until
segment I/O parity is restored. `ttfs_quantized` and `lif` still use hybrid HCM.

`LoihiSimulationStep` is not an accuracy step.  It selects one deterministic
test sample (`loihi_parity_sample_index`, default `0`), builds an HCM
`RunRecord`, replays each neural segment through Lava with
`LavaLoihiRunner.run_segments_from_reference()`, and fails the pipeline with a
localized spike-record diff if segment inputs, per-core inputs/outputs, or
segment outputs diverge.

### LIF Adaptation and cycle-accurate training

`LIFAdaptationStep` runs only when `spiking_mode == "lif"`.  `LIFAdaptationTuner`
ramps `LIFBlendActivation` from ReLU-like teacher to `LIFActivation`.  When
`cycle_accurate_lif_forward` is true, installs `_CycleAccurateForward` on
`model.forward` (picklable for pipeline cache) calling `run_cycle_accurate` with
`simulation_steps` as `T`.  The wrapper is retained after the step when
cycle-accurate mode stays enabled.

### SANA-FE simulation

`SanafeSimulationStep` loops `sanafe_sample_count` samples.  `SanafeRunner` uses float64 segment input assembly, `ChipLatency`-aligned per-core
active windows, and simulation length `T + max_latency`.  HCM reference uses
`forward_with_recording()`.  With `sanafe_parity_check`, compares
`SanafeRunRecord.to_hcm_subset()` to HCM `RunRecord` via `compare_records`.

## Exported API (\_\_init\_\_.py)

All step classes are re-exported for convenient access.

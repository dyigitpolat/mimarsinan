# pipelining/pipeline_steps/ -- Individual Pipeline Step Implementations

Each file implements one `PipelineStep` subclass corresponding to a stage
in the deployment pipeline.

## Key Components

| File | Step Class | Pipeline Phase |
|------|-----------|----------------|
| `activation_utils.py` | `has_non_relu_activations`, `RELU_COMPATIBLE_TYPES` | Shared helper for activation steps |
| `architecture_search_step.py` | `ArchitectureSearchStep` | Configuration |
| `model_configuration_step.py` | `ModelConfigurationStep` | Configuration |
| `model_building_step.py` | `ModelBuildingStep` | Model construction |
| `pretraining_step.py` | `PretrainingStep` | Training |
| `activation_analysis_step.py` | `ActivationAnalysisStep` | Quantization prep |
| `activation_adaptation_step.py` | `ActivationAdaptationStep` | Activation adaptation (always runs: gradual ReLU replacement via ActivationAdaptationTuner) |
| `clamp_adaptation_step.py` | `ClampAdaptationStep` | Quantization (runs only when activation_quantization is True) |
| `activation_shift_step.py` | `ActivationShiftStep` | Quantization (bakes shift into bias for rate-coded; TTFS shift compensation deferred to SoftCoreMappingStep) |
| `activation_quantization_step.py` | `ActivationQuantizationStep` | Quantization |
| `weight_quantization_step.py` | `WeightQuantizationStep` | Quantization |
| `quantization_verification_step.py` | `QuantizationVerificationStep` | Verification |
| `normalization_fusion_step.py` | `NormalizationFusionStep` | Optimization |
| `soft_core_mapping_step.py` | `SoftCoreMappingStep` | Mapping (computes per-source input scales via `compute_per_source_scales`; adds TTFS shift compensation; scales `hardware_bias` by the quantization factor during weight quantization) |
| `core_quantization_verification_step.py` | `CoreQuantizationVerificationStep` | Verification |
| `core_flow_tuning_step.py` | `CoreFlowTuningStep` | Tuning |
| `hard_core_mapping_step.py` | `HardCoreMappingStep` | Mapping |
| `simulation_step.py` | `SimulationStep` | Verification |
| `torch_mapping_step.py` | `TorchMappingStep` | Model conversion (torch_* types) |
| `weight_preloading_step.py` | `WeightPreloadingStep` | Load pretrained weights (replaces Pretraining) |

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

**Activation Adaptation** always runs immediately after Activation Analysis in
all pipeline configurations: it replaces non-ReLU chip-targeted bases (GELU,
LeakyReLU) with ReLU when any exist; it does not apply activation_scales or
set clamp_rate. When `activation_quantization` is True or spiking is TTFS,
**Clamp Adaptation** runs next (applies scales and trains with ClampDecorator
for quantization / TTFS saturation). Shared logic for detecting non-ReLU
activations lives in `activation_utils.py` (`has_non_relu_activations`,
`RELU_COMPATIBLE_TYPES`). Non-chip-supported perceptrons (e.g. Identity) are
already excluded from `get_perceptrons()` via `is_chip_supported_activation`.

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
must also be multiplied by `s` — otherwise the bias contribution is attenuated
by a factor of `1/s` (typically ~1/127 for 8-bit).  `SoftCoreMappingStep`
applies this scaling to `node.hardware_bias` for every NeuralCore that has one,
covering both owned-weight FC layers and bank-backed Conv2D layers.

Note: the legacy always-on axon row (used when `hardware_bias=False`) is immune
to this issue because the bias row lives inside `core_matrix` and is scaled
automatically together with the weights.

## Exported API (\_\_init\_\_.py)

All step classes are re-exported for convenient access.

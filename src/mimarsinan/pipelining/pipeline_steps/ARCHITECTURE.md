# pipelining/pipeline_steps/ -- Individual Pipeline Step Implementations

Each file implements one `PipelineStep` subclass corresponding to a stage
in the deployment pipeline.

## Key Components

| File | Step Class | Pipeline Phase |
|------|-----------|----------------|
| `architecture_search_step.py` | `ArchitectureSearchStep` | Configuration |
| `model_configuration_step.py` | `ModelConfigurationStep` | Configuration |
| `model_building_step.py` | `ModelBuildingStep` | Model construction |
| `pretraining_step.py` | `PretrainingStep` | Training |
| `activation_analysis_step.py` | `ActivationAnalysisStep` | Quantization prep |
| `clamp_adaptation_step.py` | `ClampAdaptationStep` | Quantization |
| `input_activation_analysis_step.py` | `InputActivationAnalysisStep` | Quantization prep |
| `activation_shift_step.py` | `ActivationShiftStep` | Quantization |
| `activation_quantization_step.py` | `ActivationQuantizationStep` | Quantization |
| `weight_quantization_step.py` | `WeightQuantizationStep` | Quantization |
| `quantization_verification_step.py` | `QuantizationVerificationStep` | Verification |
| `normalization_fusion_step.py` | `NormalizationFusionStep` | Optimization |
| `soft_core_mapping_step.py` | `SoftCoreMappingStep` | Mapping |
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

## Exported API (\_\_init\_\_.py)

All step classes are re-exported for convenient access.

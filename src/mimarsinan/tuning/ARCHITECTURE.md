# tuning/ -- Training-Aware Tuning Subsystem

Manages the progressive application of activation and weight transformations
while maintaining model accuracy through smooth adaptation.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `adaptation_manager.py` | `AdaptationManager` | Manages decorator rates (clamp, shift, quantization); for TTFS omits standalone shift decorator but nests shift inside QuantizeDecorator |
| `smart_smooth_adaptation.py` | `SmartSmoothAdaptation` | Framework for gradually applying transformations with accuracy recovery |
| `basic_smooth_adaptation.py` | `BasicSmoothAdaptation` | Basic step-based adaptation (base for SmartSmoothAdaptation) |
| `basic_interpolation.py` | `BasicInterpolation` | Linear interpolation utilities for adaptation schedules |
| `adaptation_target_adjuster.py` | `AdaptationTargetAdjuster` | Dynamically adjusts accuracy targets during adaptation |
| `learning_rate_explorer.py` | `LearningRateExplorer` | Binary search for optimal learning rate during tuning |
| `shift_calculation.py` | `calculate_activation_shift` | Computes activation shift amounts for quantization alignment |

### Subdirectory

| Directory | Purpose |
|-----------|---------|
| `tuners/` | Concrete tuner implementations for specific transformations |

## Dependencies

- **Internal**: `models.layers` (all decorator types), `model_training` (trainers).
- **External**: `torch`, `copy`.

## Dependents

- `pipelining.pipeline_steps` imports `AdaptationManager` (model building),
  `calculate_activation_shift` (activation shift step), and tuners.

## Exported API (\_\_init\_\_.py)

`AdaptationManager`, `SmartSmoothAdaptation`, `LearningRateExplorer`,
`calculate_activation_shift`.

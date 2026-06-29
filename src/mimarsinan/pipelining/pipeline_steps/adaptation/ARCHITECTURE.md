# pipeline_steps/adaptation/ — Adaptation and Analysis Steps

Activation analysis, activation preconditioning, clamp/LIF/noise/pruning adaptation, activation shift.

Cycle-accurate LIF and `ttfs_cycle_based` pipelines run Activation Adaptation,
Clamp, Activation Shifting, and Activation Quantization before their dedicated
LIF / TTFS-cycle tuning step.

| File | Step class |
|------|------------|
| `activation_analysis_step.py` | `ActivationAnalysisStep` |
| `activation_adaptation_step.py` | `ActivationAdaptationStep` |
| `clamp_adaptation_step.py` | `ClampAdaptationStep` |
| `activation_shift_step.py` | `ActivationShiftStep` |
| `pruning_adaptation_step.py` | `PruningAdaptationStep` |
| `lif_adaptation_step.py` | `LIFAdaptationStep` |
| `ttfs_cycle_adaptation_step.py` | `TTFSCycleAdaptationStep` (ttfs_cycle_based only; always runs for that mode; after Activation Quantization, before Weight Quantization) |
| `noise_adaptation_step.py` | `NoiseAdaptationStep` |

Shared helpers: `pipelining.pipeline_steps.activation_utils`.

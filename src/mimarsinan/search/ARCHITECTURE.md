# search/ -- Architecture Search Subsystem

Provides a clean, protocol-based multi-objective architecture search framework
with pluggable optimizers and problem definitions.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `problem.py` | `SearchProblem` | Protocol defining the search problem interface |
| `results.py` | `ObjectiveSpec`, `Candidate`, `SearchResult` | Result types and Pareto front utilities |
| `basic_architecture_searcher.py` | `BasicArchitectureSearcher` | Legacy searcher (unused) |
| `basic_configuration_sampler.py` | `BasicConfigurationSampler` | Base configuration sampler |
| `mlp_mixer_configuration_sampler.py` | `MLP_Mixer_ConfigurationSampler` | MLP-Mixer-specific config sampler |
| `mlp_mixer_searcher.py` | `MLP_Mixer_Searcher` | Legacy MLP-Mixer searcher |
| `small_step_evaluator.py` | `SmallStepEvaluator` | Evaluates candidates with short training |
| `te_nas_evaluator.py` | `TE_NAS_Evaluator` | Training-free NAS evaluator using NTK |
| `patch_borders.py` | `get_region_borders` | Patch border computation utility |

### Subdirectories

| Directory | Purpose |
|-----------|---------|
| `optimizers/` | Optimizer implementations (NSGA-II, Kedi LLM, Sampler) |
| `evaluators/` | Accuracy evaluators for search candidates |
| `problems/` | Concrete search problem implementations |
| `multi_metric/` | Multi-metric search utilities |

## Dependencies

- **Internal**: `model_training` (trainers), `data_handling` (data loaders/providers), `model_evaluation` (NTK), `mapping.layout` (layout estimation).
- **External**: `pymoo`, `torch`, `numpy`.

## Dependents

- `pipelining.pipeline_steps` (architecture search step, model configuration step).

## Exported API (\_\_init\_\_.py)

`SearchProblem`, `ObjectiveSpec`, `Candidate`, `SearchResult`.
Optimizers are accessed via `search.optimizers` subpackage.

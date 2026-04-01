# search/ -- Architecture Search Subsystem

Provides a clean, protocol-based multi-objective architecture search framework
with pluggable optimizers and problem definitions.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `problem.py` | `SearchProblem`, `ValidationResult` | Protocol defining the search problem interface; `ValidationResult` carries rich error info from `validate_detailed()` |
| `results.py` | `ObjectiveSpec`, `ALL_OBJECTIVES`, `default_objectives_for_mode`, `Candidate`, `SearchResult` | Canonical multi-objective catalogue (includes `fragmentation_pct` for packing dead zones), defaults per search mode, and Pareto utilities |
| `te_nas_evaluator.py` | `TE_NAS_Evaluator` | Training-free NAS evaluator using NTK |
| `patch_borders.py` | `get_region_borders` | Patch border computation utility |

### Subdirectories

| Directory | Purpose |
|-----------|---------|
| `optimizers/` | Optimizer implementations (NSGA-II, Agentic Evolution LLM) |
| `evaluators/` | Accuracy evaluators for search candidates |
| `problems/` | Concrete search problem implementations |
| `multi_metric/` | Multi-metric search utilities |

## Dependencies

- **Internal**: `model_training` (trainers), `data_handling` (data loaders/providers), `model_evaluation` (NTK), `mapping.layout` (layout estimation).
- **External**: `pymoo`, `torch`, `numpy`.

## Dependents

- `pipelining.pipeline_steps` (architecture search step, model configuration step).

## Exported API (\_\_init\_\_.py)

`SearchProblem`, `ValidationResult`, `ObjectiveSpec`, `Candidate`, `SearchResult`.
Optimizers are accessed via `search.optimizers` subpackage.

# search/optimizers/ -- Search Optimizer Implementations

Pluggable optimizer implementations for the search framework.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `base.py` | `SearchOptimizer` | Abstract base class for all optimizers |
| `nsga2_optimizer.py` | `NSGA2Optimizer` | Multi-objective genetic algorithm via pymoo's NSGA-II |
| `kedi_optimizer.py` | `KediOptimizer` | LLM-based optimizer using agentic reasoning (optional, requires `kedi`) |
| `kedi_optimizer_support.py` | (support utilities) | Prompt templates and analysis helpers for Kedi |
| `sampler_optimizer.py` | `SamplerOptimizer` | Simple sampler-based optimizer with feedback |
| `test_kedi_optimizer.py` | (test) | Unit tests for KediOptimizer |

## Dependencies

- **Internal**: `search.problem`, `search.results`, `search.problems.encoded_problem`.
- **External**: `pymoo`, `numpy`, `kedi` (optional).

## Dependents

- `pipelining.pipeline_steps.architecture_search_step` selects and uses optimizers.
- `search.mlp_mixer_searcher` uses `SamplerOptimizer`.

## Exported API (\_\_init\_\_.py)

`SearchOptimizer`, `NSGA2Optimizer`, `SamplerOptimizer`, and optionally `KediOptimizer`.

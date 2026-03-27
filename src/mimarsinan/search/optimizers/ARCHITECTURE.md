# search/optimizers/ -- Search Optimizer Implementations

Pluggable optimizer implementations for the search framework.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `base.py` | `SearchOptimizer` | Abstract base class for all optimizers |
| `nsga2_optimizer.py` | `NSGA2Optimizer` | Multi-objective genetic algorithm via pymoo's NSGA-II |
| `kedi_optimizer.py` | `KediOptimizer` | LLM-based optimizer using agentic reasoning (optional, requires `kedi`); orchestration only, prompts in `kedi_prompts` |
| `kedi_prompts.py` | `build_*_prompt`, `parse_candidates` | Prompt template builders and candidate parsing for Kedi LLM calls |
| `kedi_optimizer_support.py` | `CandidateResult`, `compute_pareto_front`, `prettify_*`, etc. | Pareto/formatting and analysis helpers for Kedi |
| `test_kedi_optimizer.py` | (test) | Unit tests for KediOptimizer |

## Dependencies

- **Internal**: `search.problem`, `search.results`, `search.problems.encoded_problem`.
- **External**: `pymoo`, `numpy`, `kedi` (optional).

## Dependents

- `pipelining.pipeline_steps.architecture_search_step` selects and uses optimizers.

## Exported API (\_\_init\_\_.py)

`SearchOptimizer`, `NSGA2Optimizer`, and optionally `KediOptimizer`.

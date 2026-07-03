# search/problems/ -- Concrete Search Problems

Provides concrete `SearchProblem` implementations for different search
scenarios.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `encoded_problem.py` | `EncodedProblem` | Protocol extending `SearchProblem` with continuous variable encoding for evolutionary optimizers |
| `joint/` | `JointArchHwProblem`, `effective_max_dims`, `json_key` | Joint architecture + hardware co-search (`problem.py`, `validate.py`, `layout_hook.py`, `evaluate.py`, `types.py`). `types.py` also holds `JointHostContract`, the annotation-only declaration of the host members the three mixins use through `self` (empty at runtime; kept in sync by `tests/unit/search/test_host_contracts.py`). Normalizes `platform_constraints` via `mapping.coalescing.normalize_coalescing_config`. Full feasibility in `validate_detailed()`; built model and HW objectives cached so `evaluate()` only runs accuracy training on validated candidates. Error contract: candidate-scoped failures (build/convert/pack/accuracy for a given config) degrade to explicit invalid/penalty results with a `logging` warning; candidate-independent failures (e.g. the fixed model of a `hardware`-mode search failing to build) propagate. |

## Dependencies

- **Internal**: `search.problem`, `search.results`, `search.evaluators`, `mapping.layout` (layout estimation), `mapping.coalescing` (canonical coalescing flag).
- **External**: `numpy`.

## Dependents

- `search.optimizers.nsga2_optimizer` uses `EncodedProblem`
- `pipelining.pipeline_steps.architecture_search_step` uses `JointArchHwProblem`

## Exported API (\_\_init\_\_.py)

`EncodedProblem`, `JointArchHwProblem`.

# search/problems/ -- Concrete Search Problems

Provides concrete `SearchProblem` implementations for different search
scenarios.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `encoded_problem.py` | `EncodedProblem` | Protocol extending `SearchProblem` with continuous variable encoding for evolutionary optimizers |
| `joint_arch_hw_problem.py` | `JointArchHwProblem`, `effective_max_dims` | Joint architecture + hardware co-search problem (model-agnostic). Normalizes `platform_constraints` via `mapping.coalescing.normalize_coalescing_config` (canonical `allow_coalescing` only). Full feasibility (model build + HW packing) is checked in `validate_detailed()`; the built model and HW objectives are cached so `evaluate()` only runs accuracy training on already-validated candidates. |

## Dependencies

- **Internal**: `search.problem`, `search.results`, `search.evaluators`, `mapping.layout` (layout estimation), `mapping.coalescing` (canonical coalescing flag).
- **External**: `numpy`.

## Dependents

- `search.optimizers.nsga2_optimizer` uses `EncodedProblem`
- `pipelining.pipeline_steps.architecture_search_step` uses `JointArchHwProblem`

## Exported API (\_\_init\_\_.py)

`EncodedProblem`, `JointArchHwProblem`.

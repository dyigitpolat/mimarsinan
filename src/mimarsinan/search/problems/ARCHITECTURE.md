# search/problems/ -- Concrete Search Problems

Provides concrete `SearchProblem` implementations for different search
scenarios.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `encoded_problem.py` | `EncodedProblem` | Protocol extending `SearchProblem` with continuous variable encoding for evolutionary optimizers |
| `joint_arch_hw_problem.py` | `JointArchHwProblem`, `effective_max_dims` | Joint architecture + hardware co-search problem (model-agnostic). Normalizes `platform_constraints` via `mapping.coalescing.normalize_coalescing_config`. Layout tiling uses `mapping.platform_constraints.resolve_platform_mapping_params` (same effective axon count as SCM/wizard). Full feasibility is checked in `validate_detailed()`; built model and HW objectives are cached so `evaluate()` only runs accuracy training on validated candidates. |

## Dependencies

- **Internal**: `search.problem`, `search.results`, `search.evaluators`, `mapping.layout` (layout estimation), `mapping.coalescing` (canonical coalescing flag).
- **External**: `numpy`.

## Dependents

- `search.optimizers.nsga2_optimizer` uses `EncodedProblem`
- `pipelining.pipeline_steps.architecture_search_step` uses `JointArchHwProblem`

## Exported API (\_\_init\_\_.py)

`EncodedProblem`, `JointArchHwProblem`.

# search/problems/ -- Concrete Search Problems

Provides concrete `SearchProblem` implementations for different search
scenarios.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `encoded_problem.py` | `EncodedProblem` | Protocol extending `SearchProblem` with continuous variable encoding for evolutionary optimizers |
| `evaluator_problem.py` | `EvaluatorProblem` | Wraps a scalar evaluator into a `SearchProblem` |
| `joint_arch_hw_problem.py` | `JointArchHwProblem` | Joint architecture + hardware co-search problem (model-agnostic) |

## Dependencies

- **Internal**: `search.problem`, `search.results`, `search.evaluators`, `mapping.layout` (layout estimation).
- **External**: `numpy`.

## Dependents

- `search.optimizers.nsga2_optimizer` uses `EncodedProblem`
- `pipelining.pipeline_steps.architecture_search_step` uses `JointArchHwProblem`
- `pipelining.pipeline_steps.model_configuration_step` uses `EvaluatorProblem`

## Exported API (\_\_init\_\_.py)

`EncodedProblem`, `EvaluatorProblem`, `JointArchHwProblem`.

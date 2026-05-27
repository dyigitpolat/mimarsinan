# search/evaluators/ -- Search Candidate Evaluators

Provides accuracy evaluators that score architecture candidates during search.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `fast_accuracy_evaluator.py` | `FastAccuracyEvaluator` | Quick accuracy estimation with limited training epochs |
| `extrapolating_accuracy_evaluator.py` | `ExtrapolatingAccuracyEvaluator` | Accuracy estimation with learning-curve extrapolation |
| `learning_curve.py` | `fit_and_extrapolate`, `soft_clip_accuracy` | Parametric curve models used by the extrapolating evaluator |

## Dependencies

- **Internal**: `data_handling` (`DataLoaderFactory`, `DataProviderFactory`).
- **External**: `torch`.

## Dependents

- `search.problems.joint` uses both evaluators.

## Exported API (\_\_init\_\_.py)

`FastAccuracyEvaluator`, `ExtrapolatingAccuracyEvaluator`.

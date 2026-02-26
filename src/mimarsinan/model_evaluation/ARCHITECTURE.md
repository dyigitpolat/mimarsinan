# model_evaluation/ -- Model Evaluation Utilities

Provides training-free evaluation metrics used by architecture search.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `te_nas_utils.py` | `get_ntk`, `recal_bn` | Neural Tangent Kernel computation for training-free NAS evaluation |

## Dependencies

- **Internal**: None.
- **External**: `torch`.

## Dependents

- `search.te_nas_evaluator` uses `get_ntk` for training-free architecture scoring.

## Exported API (\_\_init\_\_.py)

`get_ntk`, `recal_bn`.

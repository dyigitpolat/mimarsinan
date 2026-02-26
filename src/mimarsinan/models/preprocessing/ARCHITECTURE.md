# models/preprocessing/ -- Input Preprocessing

Provides input preprocessing modules applied before the main model computation.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `input_cq.py` | `InputCQ` | Input Clamp-Quantize preprocessor: normalizes inputs to [0,1] and applies quantization |

## Dependencies

- **Internal**: `models.layers` (`TransformedActivation`, `ClampDecorator`, `QuantizeDecorator`).
- **External**: `torch`.

## Dependents

- All `models.builders` import `InputCQ` to construct `Supermodel` instances.

## Exported API (\_\_init\_\_.py)

`InputCQ`.

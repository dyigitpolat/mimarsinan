# mapping/mappers/ — Mapper Hierarchy

`Mapper` subclasses that emit IR from the `ModelRepresentation` graph.

## Key modules

| File | Role |
|------|------|
| `base.py` | `Mapper` base class |
| `perceptron.py` | FC / perceptron layers |
| `conv.py` | Conv2D perceptron mappers |
| `structural.py` | Concat, stack, structural ops |
| `leading_dim.py` | Leading-dimension reshapes |

## Dependents

- `mapping.mapping_utils`, `torch_mapping`, model builders.

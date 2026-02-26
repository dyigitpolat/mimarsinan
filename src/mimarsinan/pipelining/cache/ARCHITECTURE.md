# pipelining/cache/ -- Pipeline Cache

Provides persistent key-value storage for inter-step data transfer with
pluggable serialization strategies.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `pipeline_cache.py` | `PipelineCache` | Dict-like store with save/load to disk; manages namespaced keys |
| `load_store_strategies.py` | `LoadStoreStrategy`, `BasicLoadStoreStrategy`, `TorchModelLoadStoreStrategy`, `PickleLoadStoreStrategy` | Serialization strategies: JSON, torch.save, pickle |

## Dependencies

- **Internal**: None.
- **External**: `torch`, `json`, `pickle`, `os`.

## Dependents

- `pipelining.pipeline.Pipeline` uses `PipelineCache` for all inter-step state.

## Exported API (\_\_init\_\_.py)

`PipelineCache` and all strategy classes.

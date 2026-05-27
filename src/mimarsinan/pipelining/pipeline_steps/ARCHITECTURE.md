# pipelining/pipeline_steps/ — Pipeline Step Implementations

Each submodule implements `PipelineStep` subclasses for one pipeline phase. All steps are re-exported from [`__init__.py`](__init__.py).

## Subpackages

| Directory | Doc | Phase |
|-----------|-----|-------|
| `config/` | [config/ARCHITECTURE.md](config/ARCHITECTURE.md) | Configuration, build, arch search |
| `training/` | [training/ARCHITECTURE.md](training/ARCHITECTURE.md) | Pretraining |
| `adaptation/` | [adaptation/ARCHITECTURE.md](adaptation/ARCHITECTURE.md) | Activation analysis & adaptation |
| `quantization/` | [quantization/ARCHITECTURE.md](quantization/ARCHITECTURE.md) | Quantization & fusion |
| `mapping/` | [mapping/ARCHITECTURE.md](mapping/ARCHITECTURE.md) | Soft/hard core mapping |
| `verification/` | [verification/ARCHITECTURE.md](verification/ARCHITECTURE.md) | Simulation backends |

## Shared

| File | Role |
|------|------|
| `activation_utils.py` | `has_non_relu_activations`, shared activation helpers |

## Dependencies

Nearly all domain packages: `models`, `mapping`, `tuning`, `chip_simulation`, `visualization`, `search`.

## Dependents

- `pipelining.pipelines.deployment_pipeline`

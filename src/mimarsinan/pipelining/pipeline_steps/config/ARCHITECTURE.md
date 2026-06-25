# pipeline_steps/config/ — Configuration Phase Steps

Model configuration, building, torch mapping, weight preload, and architecture search.

| File | Step class |
|------|------------|
| `architecture_search_step.py` | `ArchitectureSearchStep` |
| `model_configuration_step.py` | `ModelConfigurationStep` |
| `model_building_step.py` | `ModelBuildingStep` |
| `torch_mapping_step.py` | `TorchMappingStep` |
| `weight_preloading_step.py` | `WeightPreloadingStep` |

`WeightPreloadingStep` (the F3 pretrained regime) resolves its strategy via
`resolve_weight_strategy(plan.weight_source, builder)` at the TOP of `process()`,
before any load/fine-tune. A `weight_source='torchvision'` request for a builder
with no `get_pretrained_factory()` (any native from-scratch vehicle) therefore
raises the typed `UnsupportedPreloadError` EARLY; `run.py` catches it and records a
CLEAN `skipped`/UNSUPPORTED exit (0) instead of an opaque rc=1 mid-pipeline crash.

Re-exported from `pipelining.pipeline_steps.__init__`.

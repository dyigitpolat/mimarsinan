# pipelining/pipeline_steps/ — Pipeline Step Implementations

Each submodule implements `PipelineStep` subclasses for one pipeline phase. All steps are re-exported from [`__init__.py`](__init__.py).

**Applicability (Vector V5):** each step declares `@classmethod applies_to(plan)` — whether it belongs in the pipeline for a resolved `DeploymentPlan`. The base `PipelineStep.applies_to` returns `True`; conditional steps override with the verbatim per-flag predicate that gated their former `append` in `deployment_specs` (e.g. `ArchitectureSearchStep` ↔ `plan.search_mode != "fixed"`, `WeightPreloadingStep` ↔ `bool(plan.weight_source)`, the activation-family steps ↔ `plan.is_lif_style` / `plan.activation_quantization` / `plan.requires_ttfs_firing`, the `*QuantizationStep`s ↔ `plan.weight_quantization`). The V5 `StepPlan` registry (`core/step_plan.py`) filters an ordered registry by these predicates; backend steps stay in `BACKEND_REGISTRY`.

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

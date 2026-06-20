# pipelining/pipeline_steps/ — Pipeline Step Implementations

Each submodule implements `PipelineStep` subclasses for one pipeline phase. All steps are re-exported from [`__init__.py`](__init__.py).

**Applicability (Vector V5):** each step declares `@classmethod applies_to(plan)` — whether it belongs in the pipeline for a resolved `DeploymentPlan`. The base `PipelineStep.applies_to` returns `True`; conditional steps override with the verbatim per-flag predicate that gated their former `append` in `deployment_specs` (e.g. `ArchitectureSearchStep` ↔ `plan.search_mode != "fixed"`, `WeightPreloadingStep` ↔ `bool(plan.weight_source)`, the activation-family steps ↔ `plan.is_lif_style` / `plan.activation_quantization` / `plan.requires_ttfs_firing`, the `*QuantizationStep`s ↔ `plan.weight_quantization`). The V5 `StepPlan` registry (`core/step_plan.py`) filters an ordered registry by these predicates; backend steps stay in `BACKEND_REGISTRY`.

**Data contract (Vector V5):** each step declares its data contract as the CLASS-level constants `REQUIRES` / `PROMISES` / `UPDATES` / `CLEARS` (tuples; default `()` on the base). `__init__` reads these (`super().__init__(self.REQUIRES, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)`), and `PipelineStep.declared_contract()` exposes them so `StepPlan.validate_data_contract(plan)` can check the requires/promises DAG at ASSEMBLY time (no instantiation) and fail loud naming the missing producer. The lifted class declarations are byte-identical to the former local `__init__` lists. `TTFSCycleAdaptationStep` is the ONE step whose instance contract still extends the class one in `__init__`: it appends `activation_scales` to `requires` only when `ttfs_scale_aware_boundaries` is on — its `REQUIRES` declares the static lower bound (always satisfiable; the extra is itself promised earlier by the unconditional Activation Analysis).

**Deployment-flag reads (Vector V1):** inside `process()`, steps read deployment-decision flags (`spiking_mode`, `activation_quantization`/`weight_quantization`, `pruning`/`pruning_fraction`, `weight_source`, …) from `DeploymentPlan.of(self.pipeline).<field>` — the single resolver (`core/deployment_plan.py`) — never via a raw `config.get(<flag>)`. A grep-guard (`tests/unit/pipelining/test_deployment_plan.py::TestPipelineStepsReadThePlan`) locks this for the whole subtree.

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

# research/ -- Prototype-to-production contracts

Typed, pure-data contracts for expressing research prototypes as production
configuration overlays. This package must not become a second deployment stack;
it names vehicles, mechanisms, budgets, contexts, and measured outcomes so probe
ideas can be promoted through `run.py`, `DeploymentPlan`, and the ledger schema.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `harness.py` | `VehicleSpec`, `FixRecipe`, `BudgetSchedule`, `ExperimentContext`, `MechanismResult`, `recipe_presets`, `recipe_preset`, `normalize_retention`, `promotion_record`; `AcceptanceGate`, `MixerBudgetSchedule`, `RecipePreset`, `DiagnosticCell`, `DiagnosticManifest`, `recipe_registry`, `build_mnist_mixer_manifest`, `planned_mnist_mixer_ledger_row` | Vehicle/recipe/budget/context contracts plus named default-off recipe presets and promotion-gate summaries for retention, timing, parity, and validity. Also hosts the MNIST `mlp_mixer_core` closure contract: hard accuracy/runtime acceptance, diagnostic/control cells, mixer recipe presets, and planned ledger rows. `config_overlay()` methods and manifest helpers emit production config fragments/metadata; they do not run models or bypass the pipeline. |

## Dependencies

- **Internal**: `mimarsinan.chip_simulation.ledger_schema` for planned mixer ledger rows.
- **External**: dataclasses / typing only.

## Dependents

- Future probe CLIs and campaign generators can consume these contracts to avoid
  hardcoded vehicle/fix matrices.

## Exported API (`__init__.py`)

`VehicleSpec`, `FixRecipe`, `BudgetSchedule`, `ExperimentContext`,
`MechanismResult`, `recipe_presets`, `recipe_preset`, `normalize_retention`,
`promotion_record`, plus the MNIST mixer diagnostics API (`AcceptanceGate`,
`MixerBudgetSchedule`, `RecipePreset`, `DiagnosticCell`, `DiagnosticManifest`,
`recipe_registry`, `build_mnist_mixer_manifest`,
`planned_mnist_mixer_ledger_row`).

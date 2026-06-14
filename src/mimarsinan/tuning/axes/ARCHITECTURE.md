# tuning/axes/ -- Rate-driven adaptation axes (AdaptationAxis contract)

First-class, control-facing objects for the ANN→SNN tuning refactor. An
`AdaptationAxis` is the homotopy α-axis the tuner/driver walks from 0 (original
behavior) to 1 (full transform). It is the behavioral spec's "Transformation"
contract, renamed to avoid colliding with `mimarsinan.transformations` (stateless
transform *math*). Each axis **delegates** its math to `transformations/` and its
rate application to `tuning/perceptron_rate`; it owns only the orchestration seam.

Axes are transient per-run objects (built from config + the adaptation manager);
they are **never** stored on the model or the pickled `adaptation_manager` cache.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `adaptation_axis.py` | `AdaptationAxis` (runtime-checkable `Protocol`), `AdaptationAxisBase` | The contract: `attach`/`set_rate`/`calibrate`/`tunable_parameters`/`recovery_hooks`/`finalize`/`get_extra_state`/`set_extra_state`/`set_decision_seed`/`descriptor` + descriptors `name`/`interpolation_mode`/`monotonicity`/`is_stochastic`/`supports_smooth`. `AdaptationAxisBase` supplies benign defaults (no-op calibrate/recovery/finalize, empty tunable params) so adapters only implement what is axis-specific. |
| `manager_rate_axis.py` | `ManagerRateAxis`, `ClampAxis`, `ActQuantAxis`, `NoiseAxis`, `ActivationAdaptationAxis` | Adapters for the `AdaptationManager` rate-field family. `set_rate` delegates to `perceptron_rate.apply_manager_rate` (set one manager field, rebuild all decorator stacks) — byte-identical to `AdaptationRateTuner._apply_rate`. State carriage is the single manager float (`get/set_extra_state`). |

## Coverage plan (remaining adapters, same delegation discipline)

- `blend_axis.py` — `BlendAxis`/`LIFAxis`/`TTFSAxis`: `set_rate` → `perceptron_rate.set_blend_rate` (live `BlendActivation.rate`, no rebuild); `finalize` **delegates to the inherited `KDBlendAdaptationTuner._finalize`** (parity-critical forward-install; `test_finalize_contract` forbids reimplementing it).
- `perceptron_transform_axis.py` — `PerceptronTransformAxis`/`NAPQAxis`: closure mechanism (`_mixed_transform`); trainer passed at `set_rate` call time, never stored on the axis.
- `activation_shift_axis.py` — `ActivationShiftAxis`: one-shot (`supports_smooth=False`).
- `pruning_axis.py` — `PruningAxis`: dict-of-sets persistent state, `recovery_hooks` enforce masks during training, persistent-prune `finalize`.

## Dependencies

- **Internal**: `tuning.perceptron_rate` (SSOT rate application), `transformations/` (math, via the future blend/closure/pruning adapters).
- **External**: `torch`.

## Dependents

- (P1+) The tuner's rate application, gated behind `tuning_use_axis`; later `AdaptationDriver` (P4) consumes axes directly.

## Exported API (\_\_init\_\_.py)

`AdaptationAxis`, `AdaptationAxisBase`, `ManagerRateAxis`, `ClampAxis`,
`ActQuantAxis`, `NoiseAxis`, `ActivationAdaptationAxis`.

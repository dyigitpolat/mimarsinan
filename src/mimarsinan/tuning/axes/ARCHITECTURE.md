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
| `adaptation_axis.py` | `AdaptationAxis` (runtime-checkable `Protocol`), `AdaptationAxisBase` | The contract: `attach`/`set_rate`/`calibrate`/`tunable_parameters`/`recovery_hooks`/`finalize`/`get_extra_state`/`set_extra_state`/`set_decision_seed`/`descriptor` + descriptors `name`/`interpolation_mode`/`monotonicity`/`is_stochastic`/`supports_smooth`. `AdaptationAxisBase` supplies benign defaults so adapters only implement what is axis-specific. |
| `manager_rate_axis.py` | `ManagerRateAxis`, `ClampAxis`, `ActQuantAxis`, `NoiseAxis`, `ActivationAdaptationAxis` | `AdaptationManager` rate-field family. `set_rate` → `perceptron_rate.apply_manager_rate` (byte-identical to the inline path; see `test_perceptron_rate`). State carriage is the single manager float. Routes `AdaptationRateTuner`/`ClampTuner`/`ActivationAdaptationTuner`. |
| `blend_axis.py` | `BlendAxis`, `LIFAxis`, `TTFSAxis` | KD-blend family. `set_rate` → `perceptron_rate.set_blend_rate` (live `BlendActivation.rate`, no rebuild); state is the per-perceptron rate list. Routes `KDBlendAdaptationTuner`. `finalize` is **not** owned — the parity-critical forward-install stays on the tuner's inherited `_finalize` (`test_finalize_contract` forbids reimplementing it). |
| `perceptron_transform_axis.py` | `PerceptronTransformAxis`, `NAPQAxis` | Stochastic closure mechanism. Thin uniform `set_rate` seam over a tuner-provided `apply_fn` (the tuner owns the prev/new builders + trainer); folding the mechanism into the axis is the P4 driver refactor. Routes `NormalizationAwarePerceptronQuantizationTuner`. |
| `pruning_axis.py` | `PruningAxis` | Structured pruning. Thin seam over the tuner's mask-apply + recovery-hook callables; persistent-prune `finalize` stays on the tuner. Routes `PruningTuner`. |
| `activation_shift_axis.py` | `ActivationShiftAxis` | One-shot shift (`supports_smooth=False`); thin seam over the tuner's `_apply_shift`. Routes `ActivationShiftTuner`. |

## Routing

All tuner families route their rate application through an axis when
`config["tuning_use_axis"]` is set (default off); flag-off is byte-identical
(the seam delegates to the same SSOT / extracted callable). The flag is not yet
flipped — the manager-rate / blend families own their mechanism in the axis; the
closure / pruning / shift families present the seam over a tuner callable, so the
mechanism-into-axis move (the standalone `AdaptationDriver`, P4) is still pending.

## Dependencies

- **Internal**: `tuning.perceptron_rate` (SSOT rate application), `transformations/` (math, via the future blend/closure/pruning adapters).
- **External**: `torch`.

## Dependents

- (P1+) The tuner's rate application, gated behind `tuning_use_axis`; later `AdaptationDriver` (P4) consumes axes directly.

## Exported API (\_\_init\_\_.py)

`AdaptationAxis`, `AdaptationAxisBase`, `ManagerRateAxis`, `ClampAxis`,
`ActQuantAxis`, `NoiseAxis`, `ActivationAdaptationAxis`.

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
| `manager_rate_axis.py` | `ManagerRateAxis`, `ClampAxis`, `ActQuantAxis`, `NoiseAxis`, `ActivationAdaptationAxis` | `AdaptationManager` rate-field family. The `RateAdjustedDecorator`-backed rates (`quantization_rate`/`clamp_rate`/`activation_adaptation_rate`) build the decorator stack once (`manager.bind_rate_buffer`) and write a shared in-place `RateBuffer` per step (O(1), no rebuild — the W9 fix; output- and RNG-conformant with a full rebuild, `test_rate_buffer`); the manager field is kept in sync as a write-through so state queries never see a stale rate. `noise_rate` is `NoisyDropout`-backed (not a decorator), so `set_rate` delegates to the `perceptron_rate.apply_manager_rate` SSOT (rebuild path). `set_decision_seed` (overridden here) wires a per-device seeded `torch.Generator` into the stochastic decorators it finds (`RandomMaskAdjustmentStrategy`/`NoisyDropout`), re-wiring after each `set_rate` rebuild, so ActQuant/Noise decisions are reproducible (default generator None = bit-exact global-RNG path). Routes `AdaptationRateTuner`/`ClampTuner`/`ActivationAdaptationTuner`. |
| `blend_axis.py` | `BlendAxis`, `LIFAxis`, `TTFSAxis`, `TTFSGenuineAxis`, `GenuineBlendAxis` | KD-blend family. `set_rate` → `perceptron_rate.set_blend_rate` (live `BlendActivation.rate`, no rebuild); state is the per-perceptron rate list. Routes `KDBlendAdaptationTuner`. `finalize` is **not** owned — the parity-critical forward-install stays on the tuner's inherited `_finalize` (`test_finalize_contract` forbids reimplementing it). `TTFSGenuineAxis` (the opt-in genuine annealed ramp, `TTFSCycleAdaptationTuner._make_axis`): `set_rate` walks the same blend rate **and** anneals the spike surrogate `alpha` smooth→sharp on the geometric schedule `_alpha_for_rate(r) = alpha_min·(alpha_max/alpha_min)**r` (from `ttfs_ramp_alpha_min`/`ttfs_ramp_alpha_max`) via `perceptron_rate.set_surrogate_alpha`. `alpha` is **backward-only** (the `pre>0` Heaviside forward is unchanged), so rate=1 (`alpha_max`) is bit-identical to the deployed cascade. `GenuineBlendAxis` (the opt-in genuine teacher→cascade blend ramp, `TTFSCycleAdaptationTuner._make_axis`) is **not** a `BlendAxis` subclass: the blend is at the model OUTPUT, so `set_rate` mutates the installed `BlendedGenuineForward.rate` live (no per-perceptron carriage, no rebuild) and is a no-op when no blend forward is installed; state carriage is the scalar forward rate. |
| `perceptron_transform_axis.py` | `PerceptronTransformAxis`, `NAPQAxis` | Stochastic closure mechanism. Thin uniform `set_rate` seam over a tuner-provided `apply_fn` (the tuner owns the prev/new builders + trainer); folding the mechanism into the axis is the P4 driver refactor. Routes `NormalizationAwarePerceptronQuantizationTuner`. |
| `pruning_axis.py` | `PruningAxis` | Structured pruning. Thin seam over the tuner's mask-apply + recovery-hook callables; persistent-prune `finalize` stays on the tuner. Routes `PruningTuner`. |
| `activation_shift_axis.py` | `ActivationShiftAxis` | One-shot shift (`supports_smooth=False`); thin seam over the tuner's `_apply_shift`. Routes `ActivationShiftTuner`. |

## Routing

All tuner families route their rate application through an axis (the sole path —
graduated). The seam delegates to the same SSOT / extracted callable, so it is
byte-identical to the deleted inline path (frozen into the golden traces and
`test_axis_delegation`). The manager-rate / blend families own their mechanism in
the axis; the closure / pruning / shift families present the seam over a tuner
callable. The per-cycle control flow is now owned by `AdaptationDriver.run_cycle`
(the `_adaptation` god-method dissolved into named host phases); the fully
standalone driver constructed from the services is the remaining V6 polish.

## Dependencies

- **Internal**: `tuning.perceptron_rate` (SSOT rate application), `transformations/` (math, via the future blend/closure/pruning adapters).
- **External**: `torch`.

## Dependents

- The tuner's rate application (the sole path); the standalone `AdaptationDriver` will consume axes directly.

## Exported API (\_\_init\_\_.py)

`AdaptationAxis`, `AdaptationAxisBase`, `ManagerRateAxis`, `ClampAxis`,
`ActQuantAxis`, `NoiseAxis`, `ActivationAdaptationAxis`.

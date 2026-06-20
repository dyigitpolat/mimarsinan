# Refactor plan — collapse the adaptation flag-thicket into 3 orthogonal abstractions

## The current landscape (measured)

`TTFSCycleAdaptationTuner` (+ its base `KDBlendAdaptationTuner` and sibling
`LIFAdaptationTuner`) selects behavior through **~28 runtime flags** and **~28 distinct
if/else branch points**, with **11 mutual-exclusion/precedence rules** hand-coded in
`_configure` (lines 226–355). A *single* conceptual strategy (e.g. the genuine-blend
ramp) is **scattered across 12+ methods** — `_make_axis`, `_make_blend`, `_make_kd_loss`,
`_ramp_forward`, `_finalize_forward_for`, `_after_install_blend`, `_invalidate_lr_cache`,
`_find_lr`, `_fast_loss`, `_fast_probe`, `_post_stabilization_hook`, `_remove_forward` —
plus a flag, an axis class, and a forward class. To add a strategy you touch 8+ methods
and must re-derive the implicit dispatch; to read one you trace flags across the file.

**Three orthogonal concerns are tangled into one flat flag namespace:**

1. **Ramp strategy** — *how the model is driven 0→1*. Mutually exclusive; pick one:
   `ttfs_genuine_annealed_ramp` / `ttfs_genuine_blend_ramp` / `ttfs_staircase_ste` /
   (default value-domain proxy). Each bundles {axis, blend, ramp-forward, loss, finalize,
   fast-hooks}. The mutual exclusion ("blend wins over annealed", "STE implies annealed")
   is exactly the signature of a **polymorphic strategy** forced into booleans.
2. **Calibration** — *numerical conversion health*, largely composable/orthogonal:
   `ttfs_gain_correction[_ramp]`, `ttfs_theta_cotrain`, `ttfs_distmatch_*`,
   `ttfs_boundary_surrogate`. The one real conflict ("gain_ramp disables theta_cotrain")
   is a step-compatibility fact, not a global flag rule.
3. **Optimization driver** — *controller vs fast ladder*: `ttfs_blend_fast` /
   `ttfs_genuine_blend_fast` / `lif_blend_fast` + `_setup_fast_ladder`. "fast requires
   blend" is an artifact of the entanglement, not a real constraint.

**The codebase already has the right patterns elsewhere** — emulate them:
`spiking/segment_policies.py` (polymorphic `LifSegmentPolicy`/`AnalyticalSegmentPolicy`/
`TtfsSegmentPolicy`, driver is mode-agnostic), `chip_simulation/firing_strategy.py`
(`FiringStrategyFactory.from_config`), `deployment_contract.training_forward_kind()`
(derived getter), `ModelRegistry`. The good seams to KEEP: `_finalize_forward_for(model)`
(probe ≡ deploy by construction), `CascadeForwardInstall`, `BlendActivation`,
`perceptron_rate` SSOT, the `AdaptationAxis` contract.

## Proposed design — 3 orthogonal abstractions

### A. `RampStrategy` (polymorphic; pick one) — mirrors segment policies
One cohesive class per ramp mode, bundling everything that varies together:
```
class RampStrategy:                      # base
    def make_axis(self, tuner): ...
    def make_blend(self, tuner, old, target, rate): ...
    def ramp_forward(self, tuner, model): ...        # None | installed forward
    def make_kd_loss(self, tuner): ...
    def after_install_blend(self, tuner): ...        # strategy-specific calibration hook
    def finalize_forward(self, tuner, model): ...
    # fast-path hooks (default no-op): fast_loss, fast_probe, post_stabilization
```
Concrete: `ValueDomainProxyRamp` (default), `GenuineAnnealedRamp`, `GenuineBlendRamp`,
`StaircaseSteRamp`. Each OWNS its axis/forward/loss/numerics. The tuner holds ONE
`self._ramp` and delegates — the 28 branch points collapse into method overrides that
live next to each other inside one class. **The STE becomes simply `StaircaseSteRamp`**
(owns `_StaircaseSteKDLoss` + the genuine-forward install + the backward hedge) — no
"STE implies annealed" hack; the class *is* the unit.

### B. `CalibrationPipeline` (composable steps) — the conversion-health concern
A list of orthogonal pre-ramp steps applied in `after_install_blend`:
`GainCorrection`, `ThetaCotrain`, `DistMatch`, `BoundarySte`. Each step declares its
`compatible_with(others)` (so "gain_ramp ⊥ theta_cotrain" is local, explicit, tested —
not a global `_configure` boolean). This is also the natural home for Research-Program-3's
healthy-calibration results (a new step slots in without touching the ramp code).

### C. `OptimizationDriver` (controller vs fast) — the optimization concern
`ControllerDriver` (SmoothAdaptation) vs `FastLadderDriver` (fixed ladder + one optimizer).
Orthogonal to the ramp strategy; "fast requires blend" disappears. The STE's deployment
fix (the integration plan) is then just `StaircaseSteRamp` + `FastLadderDriver` + split-LR
+ progressive-depth — a natural composition, not a new flag cluster.

### Config collapse
~28 flat flags → `ramp_strategy: <name>` + `calibration: [<steps>]` + `fast: bool` + each
unit's own numeric params (mix, rates, steps...) namespaced under its owner. A factory
(`RampStrategy.from_config`, like `FiringStrategyFactory`) is the SINGLE place that reads
config and builds the strategy/pipeline/driver — replacing the `_configure` thicket and
all 28 branch points.

## Incremental, test-guarded plan (strangler-fig; never a big-bang)

The existing suite is the guard: 93+ TTFS/LIF tuning tests, `test_genuine_gradual_invariants.py`
(the inert@low-rate / smooth / r=1==deploy / converges contract), the golden parity tests.
**Every phase must keep them green (byte-identical) or strictly improve.**

- **P0 — Characterize.** Confirm the invariant + per-strategy tests cover each current flag
  path; add any missing characterization tests (lock current behavior before moving it).
- **P1 — Extract `RampStrategy`.** Introduce the ABC; move each flag's branch bodies into a
  concrete strategy (verbatim). `_configure` builds `self._ramp` via a factory; the M
  methods delegate to it. Keep the old flags as the factory's input (back-compat). Assert
  byte-identical via the suite. (Biggest clutter win.)
- **P2 — Extract `CalibrationPipeline`.** Move gain/theta/distmatch/boundary into steps with
  explicit `compatible_with`. Tuner applies the pipeline; delete the per-flag guards.
- **P3 — Extract `OptimizationDriver`.** Move the fast-ladder vs controller split into drivers.
- **P4 — Collapse config.** Add `ramp_strategy`/`calibration`/`fast` keys; map the legacy
  flags to them via a thin deprecation shim (one translation table, tested); delete the
  in-method flag reads. Update templates.
- **P5 — STE fast path home.** Land the integration-plan fix as `StaircaseSteRamp` +
  `FastLadderDriver` (split-LR + progressive-depth) — now a clean composition; MNIST-validate.

## Code-quality outcome
- TTFS tuner: a thin orchestrator delegating to (strategy, calibration, driver). ~28 branch
  points → method overrides co-located in 4 small strategy classes.
- Adding a strategy = one new `RampStrategy` subclass + a factory entry (no cross-method edits).
- The STE sits naturally; calibration is composable (Program-3 results drop in as steps).
- Mutual-exclusion hacks become local, explicit, tested compatibility — or vanish.
- Mirrors the codebase's own best patterns (segment policies, FiringStrategyFactory).

"""Contract-driven resolution of the TTFS-cycle adaptation flag-thicket.

``_configure`` used to read ~24 ``ttfs_*`` flags inline and hand-derive the dispatch
(ramp strategy, optimization driver, fast-ladder rung) with the precedence /
compatibility rules scattered between the reads. ``TtfsAdaptationPlan`` is the SINGLE
declarative resolver: config (+ whether the schedule is synchronized) in, a validated,
frozen plan out. The tuner sets its ``self._*`` fields from the plan, so the precedence
rules live (and are tested) in one place instead of being re-derived across the file.

The plan composes the three orthogonal abstractions: it picks the ramp strategy flags
(below), delegates the *optimization driver* (controller vs fast-ladder) rung to
:class:`OptimizationDriver`, and the *conversion-health* steps to
:class:`CalibrationPipeline` — so the adaptation trifecta (ramp × driver × calibration)
is resolved once, each concern owning its own compatibility rules.

Precedence (settled here, in order):
* genuine ramps are cascaded-only (off when ``synchronized``);
* the genuine **blend** ramp wins over the **annealed** ramp;
* the **STE** reuses the annealed install (forces ``genuine_annealed_ramp``) and is
  excluded by the blend ramp;
* ``staircase_ste_fast`` requires the STE; ``genuine_blend_fast`` requires the blend ramp;
* the **proxy** fast path is value-domain — inert under any genuine bare-target ramp or
  synchronized; ``staircase_ste_refine`` requires the proxy path;
* the fast-ladder rung is resolved by :class:`OptimizationDriver` (STE-fast → one rung
  at 1.0; else the blend/proxy ladder, with the proxy flooring the endpoint LR);
* the calibration steps + their compatibility are resolved by
  :class:`CalibrationPipeline` keyed by the (firing × sync) ``SpikingModePolicy``
  (E3: the cascaded cycle opts into conversion-health, the synchronized cycle gets the
  inert pipeline; distmatch driven by the blend ramp).
"""

from __future__ import annotations

from dataclasses import dataclass

from mimarsinan.tuning.orchestration.calibration_pipeline import CalibrationPipeline
from mimarsinan.tuning.orchestration.optimization_driver import OptimizationDriver

_DEFAULT_BLEND_FAST_RATES = [0.5, 0.75, 0.9, 0.97, 1.0]


@dataclass(frozen=True)
class TtfsAdaptationPlan:
    # ── ramp strategy (mutually exclusive; else value-domain proxy) ──
    genuine_annealed_ramp: bool
    genuine_blend_ramp: bool
    staircase_ste: bool
    genuine_bare_target_ramp: bool
    # ── optimization driver ──
    proxy_fast: bool
    genuine_blend_fast: bool
    staircase_ste_fast: bool
    staircase_ste_refine: bool
    # ── numeric params (namespaced under their owner) ──
    ste_mix: float
    ste_steps: int
    ste_w_lr: float
    ste_theta_lr: float
    ste_init_frac: float
    blend_fast_rates: list
    blend_fast_steps_per_rate: int
    fast_stabilize_steps: int
    blend_fast_lr_eta_min: float
    # ── resolved optimization driver (controller vs fast-ladder rung) ──
    driver: OptimizationDriver
    # ── resolved conversion-health calibration steps ──
    calibration: CalibrationPipeline

    @property
    def fast_ladder_enabled(self) -> bool:
        return self.driver.fast_ladder

    @property
    def fast_ladder_rates(self) -> list:
        return self.driver.fast_ladder_rates

    @property
    def fast_ladder_steps_per_rate(self) -> int:
        return self.driver.fast_ladder_steps_per_rate

    @property
    def fast_ladder_eta_min_factor(self) -> float:
        return self.driver.fast_ladder_eta_min_factor

    @classmethod
    def resolve(
        cls, config, *, synchronized: bool, optimization_driver: str | None = None,
    ) -> "TtfsAdaptationPlan":
        get = config.get

        annealed = bool(get("ttfs_genuine_annealed_ramp", False)) and not synchronized
        blend = bool(get("ttfs_genuine_blend_ramp", False)) and not synchronized
        if blend:
            annealed = False  # blend wins over annealed
        ste = (
            bool(get("ttfs_staircase_ste", False))
            and not synchronized and not blend
        )
        if ste:
            annealed = True  # the STE reuses the annealed cascade-forward install
        bare = annealed or blend

        # EF1: the pipeline-wide `optimization_driver` axis is the controller-vs-fast
        # GATE over the three-way fast fork; the per-family flag below still selects
        # WHICH fast variant runs. `None` (back-compat for direct callers) derives the
        # gate from the legacy fast flags so the resolution stays byte-identical (those
        # flags also feed DeploymentPlan.optimization_driver). An explicit `controller`
        # vetoes every fast selector even if a legacy fast flag is set.
        from mimarsinan.pipelining.core.deployment_plan import (
            OPTIMIZATION_DRIVER_FAST,
            resolve_optimization_driver,
        )

        if optimization_driver is None:
            axis = resolve_optimization_driver(config)
        else:
            axis = str(optimization_driver).lower()
        fast_enabled = axis == OPTIMIZATION_DRIVER_FAST

        ste_fast = (
            fast_enabled and ste and bool(get("ttfs_staircase_ste_fast", False))
        )
        blend_fast = (
            fast_enabled and blend and bool(get("ttfs_genuine_blend_fast", False))
        )
        proxy_fast = (
            fast_enabled
            and bool(get("ttfs_blend_fast", False))
            and not bare and not synchronized
        )
        ste_refine = proxy_fast and bool(get("ttfs_blend_fast_ste_refine", False))

        ste_steps = max(1, int(get("ttfs_ste_steps", 1000)))
        blend_fast_rates = [
            float(r) for r in get("ttfs_blend_fast_rates", _DEFAULT_BLEND_FAST_RATES)
        ]
        blend_fast_steps = int(get("ttfs_blend_fast_steps_per_rate", 120))
        eta_min = float(get("ttfs_blend_fast_lr_eta_min", 0.1))

        # The optimization-driver concern (controller vs fast-ladder rung) owns the
        # rung derivation; the precedence above settles which fast selector is active.
        driver = OptimizationDriver.resolve(
            staircase_ste_fast=ste_fast,
            genuine_blend_fast=blend_fast,
            proxy_fast=proxy_fast,
            ste_steps=ste_steps,
            blend_fast_rates=blend_fast_rates,
            blend_fast_steps_per_rate=blend_fast_steps,
            blend_fast_lr_eta_min=eta_min,
        )
        # The conversion-health concern (E3): the ENABLE is the (firing × sync)
        # decision owned by the SpikingModePolicy, resolved off the `synchronized`
        # cycle key (cascaded → opts in, synchronized → inert). `resolve` is the
        # boolean alias for that policy; distmatch is owned by the genuine-blend ramp.
        calibration = CalibrationPipeline.resolve(
            config, synchronized=synchronized, distmatch_driven=blend,
        )

        return cls(
            genuine_annealed_ramp=annealed,
            genuine_blend_ramp=blend,
            staircase_ste=ste,
            genuine_bare_target_ramp=bare,
            proxy_fast=proxy_fast,
            genuine_blend_fast=blend_fast,
            staircase_ste_fast=ste_fast,
            staircase_ste_refine=ste_refine,
            ste_mix=float(get("ttfs_ste_mix", 0.5)),
            ste_steps=ste_steps,
            ste_w_lr=float(get("ttfs_ste_w_lr", 2e-3)),
            ste_theta_lr=float(get("ttfs_ste_theta_lr", 5e-2)),
            ste_init_frac=float(get("ttfs_ste_init_frac", 1.0 / 3.0)),
            blend_fast_rates=blend_fast_rates,
            blend_fast_steps_per_rate=blend_fast_steps,
            fast_stabilize_steps=int(get("ttfs_blend_fast_stabilize_steps", 0)),
            blend_fast_lr_eta_min=eta_min,
            driver=driver,
            calibration=calibration,
        )

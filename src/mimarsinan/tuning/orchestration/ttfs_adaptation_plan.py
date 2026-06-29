"""Contract-driven resolution of the TTFS-cycle adaptation flag-thicket.

``_configure`` used to read the ``ttfs_*`` flags inline and hand-derive the dispatch
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
* the genuine **blend** ramp is cascaded-only (off when ``synchronized``);
* synchronized QAT is a separate default-off opt-in that may use the fast
  deployed-staircase proxy path, but it does not enable the cascade forward;
* ``genuine_blend_fast`` requires the blend ramp;
* the **proxy** fast path is value-domain — inert under the genuine blend ramp or
  synchronized (unless synchronized QAT opts it in);
* the fast-ladder rung is resolved by :class:`OptimizationDriver` (the blend/proxy
  ladder, with the proxy flooring the endpoint LR);
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
    # ── ramp strategy (genuine blend ramp; else value-domain proxy) ──
    sync_genuine_qat: bool
    genuine_blend_ramp: bool
    # ── optimization driver ──
    proxy_fast: bool
    genuine_blend_fast: bool
    # ── numeric params (namespaced under their owner) ──
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
        cls,
        config,
        *,
        synchronized: bool,
        optimization_driver: str | None = None,
        calibration_resolver=None,
    ) -> "TtfsAdaptationPlan":
        get = config.get

        sync_genuine_qat = bool(get("ttfs_sync_genuine_qat", False)) and synchronized
        cascade_genuine_allowed = not synchronized
        blend = bool(get("ttfs_genuine_blend_ramp", False)) and cascade_genuine_allowed

        # EF1: the pipeline-wide `optimization_driver` axis is the controller-vs-fast
        # GATE over the fast fork; the per-family flag below still selects WHICH fast
        # variant runs. `None` (back-compat for direct callers) derives the gate from
        # the legacy fast flags so the resolution stays byte-identical (those flags also
        # feed DeploymentPlan.optimization_driver). An explicit `controller` vetoes every
        # fast selector even if a legacy fast flag is set.
        from mimarsinan.pipelining.core.deployment_plan import (
            OPTIMIZATION_DRIVER_FAST,
            resolve_optimization_driver,
        )

        if optimization_driver is None:
            axis = resolve_optimization_driver(config)
        else:
            axis = str(optimization_driver).lower()
        fast_enabled = axis == OPTIMIZATION_DRIVER_FAST

        blend_fast = (
            fast_enabled and blend and bool(get("ttfs_genuine_blend_fast", False))
        )
        proxy_fast = (
            fast_enabled
            and bool(get("ttfs_blend_fast", False))
            and not blend
            and (not synchronized or sync_genuine_qat)
        )

        blend_fast_rates = [
            float(r) for r in get("ttfs_blend_fast_rates", _DEFAULT_BLEND_FAST_RATES)
        ]
        blend_fast_steps = int(get("ttfs_blend_fast_steps_per_rate", 120))
        eta_min = float(get("ttfs_blend_fast_lr_eta_min", 0.1))

        # The optimization-driver concern (controller vs fast-ladder rung) owns the
        # rung derivation; the precedence above settles which fast selector is active.
        driver = OptimizationDriver.resolve(
            genuine_blend_fast=blend_fast,
            proxy_fast=proxy_fast,
            blend_fast_rates=blend_fast_rates,
            blend_fast_steps_per_rate=blend_fast_steps,
            blend_fast_lr_eta_min=eta_min,
        )
        # The conversion-health concern (E3): the ENABLE is the (firing × sync)
        # decision owned by the SpikingModePolicy, resolved off the `synchronized`
        # cycle key (cascaded → opts in, synchronized → inert). distmatch is owned by
        # the genuine-blend ramp. EF3: `calibration_resolver` (default None) is the
        # injected CONTRACT seam — the live tuner passes the contract's
        # `calibration_pipeline`, so the resolution is keyed by the (firing × sync)
        # contract, not the tuner's boolean. None ⇒ the raw `CalibrationPipeline.resolve`
        # boolean alias (back-compat for direct callers) — bit-exact, since the contract
        # keys identically on cascaded-vs-synchronized.
        resolver = calibration_resolver or CalibrationPipeline.resolve
        calibration = resolver(
            config, synchronized=synchronized, distmatch_driven=blend,
        )

        return cls(
            sync_genuine_qat=sync_genuine_qat,
            genuine_blend_ramp=blend,
            proxy_fast=proxy_fast,
            genuine_blend_fast=blend_fast,
            blend_fast_rates=blend_fast_rates,
            blend_fast_steps_per_rate=blend_fast_steps,
            fast_stabilize_steps=int(get("ttfs_blend_fast_stabilize_steps", 0)),
            blend_fast_lr_eta_min=eta_min,
            driver=driver,
            calibration=calibration,
        )

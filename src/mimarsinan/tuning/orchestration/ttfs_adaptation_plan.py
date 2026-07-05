"""Contract-driven resolution of the TTFS-cycle adaptation flag-thicket."""

from __future__ import annotations

from dataclasses import dataclass

from mimarsinan.tuning.orchestration.calibration_pipeline import CalibrationPipeline
from mimarsinan.tuning.orchestration.optimization_driver import OptimizationDriver

_DEFAULT_BLEND_FAST_RATES = [0.5, 0.75, 0.9, 0.97, 1.0]


@dataclass(frozen=True)
class TtfsAdaptationPlan:
    sync_genuine_qat: bool
    genuine_blend_ramp: bool
    proxy_fast: bool
    genuine_blend_fast: bool
    blend_fast_rates: list
    blend_fast_steps_per_rate: int
    endpoint_recovery_steps: int
    blend_fast_lr_eta_min: float
    driver: OptimizationDriver
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

        # Lazy: deployment_plan pulls chip_simulation, which cannot be imported at
        # this module's top without a circular import.
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

        driver = OptimizationDriver.resolve(
            genuine_blend_fast=blend_fast,
            proxy_fast=proxy_fast,
            blend_fast_rates=blend_fast_rates,
            blend_fast_steps_per_rate=blend_fast_steps,
            blend_fast_lr_eta_min=eta_min,
        )
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
            endpoint_recovery_steps=int(get("endpoint_recovery_steps", 0)),
            blend_fast_lr_eta_min=eta_min,
            driver=driver,
            calibration=calibration,
        )

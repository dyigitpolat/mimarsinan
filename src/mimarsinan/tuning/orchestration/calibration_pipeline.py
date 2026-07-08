"""Composable conversion-health calibration, keyed by the (firing × sync) cell."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mimarsinan.common.workload_profile import ResolvedWorkloadProfile

_GENERIC_DISTMATCH_BIAS_ITERS = 15
_GENERIC_DISTMATCH_CAL_BATCHES = 8


def encoder_scale_pin(config) -> float | None:
    """The deployed encoder decode scale when the plan pins it, else ``None``.

    Cascaded TTFS deploys the subsumed encoder at ``input_data_scale`` (the FT
    rebuild enforces it), so QAT must train under the same pin — P1'
    trained-parameter conservation (§6b contract-2); other modes carry theta_enc
    through mapping and keep the measured quantile.
    """
    # Lazy: chip_simulation has a fragile import cycle; a top-level import breaks
    # when this module loads before chip_simulation finishes initializing.
    from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan

    if DeploymentPlan.resolve(config).is_cascaded_ttfs:
        return 1.0
    return None


@dataclass(frozen=True)
class CalibrationPipeline:
    """The resolved, composable calibration steps for a TTFS-cycle adaptation run."""

    gain_cold: bool
    gain_ramp: bool
    theta_cotrain: bool
    distmatch: bool
    boundary_ste: bool
    boundary_surrogate_temp: float | None
    gain_rule: str
    gain_c: float
    distmatch_quantile: float
    distmatch_bias_iters: int
    distmatch_bias_eta: float
    distmatch_cal_batches: int

    @property
    def gain_active(self) -> bool:
        """Whether either gain-correction mode is active."""
        return self.gain_cold or self.gain_ramp

    @classmethod
    def inert(cls) -> "CalibrationPipeline":
        """The all-off pipeline a cell with no conversion-health calibration
        receives (byte-identical to the historical ``synchronized`` path)."""
        return cls(
            gain_cold=False,
            gain_ramp=False,
            theta_cotrain=False,
            distmatch=False,
            boundary_ste=False,
            boundary_surrogate_temp=None,
            gain_rule="relative",
            gain_c=1.9,
            distmatch_quantile=0.99,
            distmatch_bias_iters=_GENERIC_DISTMATCH_BIAS_ITERS,
            distmatch_bias_eta=0.7,
            distmatch_cal_batches=_GENERIC_DISTMATCH_CAL_BATCHES,
        )

    @classmethod
    def for_mode(
        cls, config, *, mode_policy: Any, distmatch_driven: bool = False,
    ) -> "CalibrationPipeline":
        """Resolve the active calibration steps for a (firing × sync) cell.

        A cell whose ``does_conversion_health_calibration`` is False gets the inert
        pipeline; a cell that opts in resolves the gain / theta / distmatch /
        boundary steps from config. ``distmatch_driven`` is whether the genuine-blend
        ramp owns the teacher-distribution matching.
        """
        if not bool(getattr(mode_policy, "does_conversion_health_calibration", False)):
            return cls.inert()
        return cls._resolve_active(config, distmatch_driven=distmatch_driven)

    @classmethod
    def resolve(
        cls, config, *, synchronized: bool, distmatch_driven: bool,
    ) -> "CalibrationPipeline":
        """Back-compat alias mapping the ``synchronized`` boolean to the
        cascaded/inert policy, then delegating to :meth:`for_mode`."""
        # Lazy: chip_simulation has a fragile import cycle; a top-level import breaks
        # when this module loads before chip_simulation finishes initializing.
        from mimarsinan.chip_simulation.spiking_mode_policy import (
            TtfsCascadeModePolicy,
            TtfsSyncCycleModePolicy,
        )

        policy = (
            TtfsSyncCycleModePolicy("ttfs_cycle_based", "synchronized")
            if synchronized
            else TtfsCascadeModePolicy("ttfs_cycle_based", "cascaded")
        )
        return cls.for_mode(
            config, mode_policy=policy, distmatch_driven=distmatch_driven,
        )

    @classmethod
    def _resolve_active(
        cls, config, *, distmatch_driven: bool,
    ) -> "CalibrationPipeline":
        """Resolve the active steps + compatibility for a cell that opts into
        conversion-health calibration (the cascaded cycle today)."""
        get = config.get
        calibration = ResolvedWorkloadProfile.from_config(config).calibration
        bias_iters_default = (
            _GENERIC_DISTMATCH_BIAS_ITERS
            if calibration.distmatch_bias_iters is None
            else int(calibration.distmatch_bias_iters)
        )
        cal_batches = (
            _GENERIC_DISTMATCH_CAL_BATCHES
            if calibration.distmatch_cal_batches is None
            else int(calibration.distmatch_cal_batches)
        )

        gain_ramp = bool(get("ttfs_gain_correction_ramp", False))
        gain_cold = bool(get("ttfs_gain_correction", False)) and not gain_ramp

        theta_cotrain = bool(get("ttfs_theta_cotrain", False)) and not gain_ramp

        boundary_ste = bool(get("ttfs_boundary_surrogate", False))
        boundary_temp = (
            float(get("ttfs_boundary_surrogate_temp", 1.0)) if boundary_ste else None
        )

        return cls(
            gain_cold=gain_cold,
            gain_ramp=gain_ramp,
            theta_cotrain=theta_cotrain,
            distmatch=bool(distmatch_driven),
            boundary_ste=boundary_ste,
            boundary_surrogate_temp=boundary_temp,
            gain_rule=str(get("ttfs_gain_correction_rule", "relative")),
            gain_c=float(get("ttfs_gain_correction_c", 1.9)),
            distmatch_quantile=float(get("ttfs_distmatch_quantile", 0.99)),
            distmatch_bias_iters=int(get("ttfs_distmatch_bias_iters", bias_iters_default)),
            distmatch_bias_eta=float(get("ttfs_distmatch_bias_eta", 0.7)),
            distmatch_cal_batches=cal_batches,
        )

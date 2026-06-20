"""Composable conversion-health calibration, keyed by the (firing × sync) cell.

The four *calibration* concerns (numerical conversion health, largely orthogonal to
the ramp strategy) used to be resolved inline in the TTFS tuner ``_configure`` with
per-flag guards interleaved between the reads:

* **gain-correction** — invert the cascade's depth-attenuation; a COLD one-shot trim
  (``ttfs_gain_correction``) or a rate-gated RAMP (``ttfs_gain_correction_ramp``);
* **theta-cotrain** — promote each channel's ``activation_scale`` to a trainable param;
* **distmatch** — DFQ-match the deployed cascade to the teacher distribution (driven
  by the genuine-blend ramp);
* **boundary-STE** — un-sever the genuine backward at offload boundaries.

Their **compatibility rules** were scattered global ``if``s (theta gated
``and not self._gain_ramp``; gain-ramp winning over gain-cold; every step gated
``and not self._synchronized``). ``CalibrationPipeline`` is the SINGLE declarative
resolver: config (+ the (firing × sync) ``SpikingModePolicy``, + whether the
genuine-blend ramp drives distmatch) in, a validated frozen plan of which steps are
active out, with each step's ``compatible_with`` stated and tested in one place. The
tuner binds the plan to its ``self._*`` fields; the parity-critical *application*
order stays in the tuner (gain trim before node build, theta promotion after every
scalar-theta calibration).

E3 (Fix A) hoists the RESOLUTION off the ``ttfs_*`` names onto the contract: the
conversion-health *enable* is a (firing × sync) decision owned by the
``SpikingModePolicy`` (``does_conversion_health_calibration``), not the TTFS tuner.
``for_mode`` is the pipeline-wide resolver EVERY conversion tuner consumes — LIF /
analytical / synchronized resolve an inert pipeline (all steps off ⇒ byte-identical),
and the analytical chain can receive conversion-health steps the day a policy opts in.
``resolve(synchronized=…)`` is the back-compat alias that maps the boolean to the
cascade/inert policy.

Compatibility rules (settled here):

* the conversion-health steps run iff the (firing × sync) policy opts in
  (``does_conversion_health_calibration``) — today only the cascaded cycle; inert for
  LIF / analytical / synchronized;
* gain-correction has two MUTUALLY EXCLUSIVE modes, the RAMP winning over the COLD
  trim when both flags are set;
* theta-cotrain is ``compatible_with`` everything EXCEPT the gain RAMP (both manage
  ``activation_scale``; the per-depth scalar ramp would clobber the per-channel param,
  so the gain ramp wins and theta-cotrain is disabled);
* distmatch is active iff the genuine-blend ramp drives it (the ramp owns the call).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CalibrationPipeline:
    """The resolved, composable calibration steps for a TTFS-cycle adaptation run."""

    # ── which steps are active (cascaded-only; compatibility applied) ──
    gain_cold: bool
    gain_ramp: bool
    theta_cotrain: bool
    distmatch: bool
    boundary_ste: bool
    # ── boundary-STE temperature (None = the historical severed backward) ──
    boundary_surrogate_temp: float | None
    # ── gain-correction numeric params (shared by both modes) ──
    gain_rule: str
    gain_c: float
    # ── distmatch numeric params ──
    distmatch_quantile: float
    distmatch_bias_iters: int
    distmatch_bias_eta: float

    @property
    def gain_active(self) -> bool:
        """Whether either gain-correction mode is active."""
        return self.gain_cold or self.gain_ramp

    @classmethod
    def inert(cls) -> "CalibrationPipeline":
        """The all-off pipeline a (firing × sync) cell with no conversion-health
        calibration receives — byte-identical to the historical ``synchronized``
        path. Every conversion tuner can hold one without changing behavior."""
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
            distmatch_bias_iters=15,
            distmatch_bias_eta=0.7,
        )

    @classmethod
    def for_mode(
        cls, config, *, mode_policy: Any, distmatch_driven: bool = False,
    ) -> "CalibrationPipeline":
        """Resolve the active calibration steps for a (firing × sync) cell.

        The pipeline-wide resolver EVERY conversion tuner consumes: the ENABLE is a
        (firing × sync) decision owned by ``mode_policy`` — a cell whose
        ``does_conversion_health_calibration`` is False (LIF / analytical /
        synchronized today) gets the inert pipeline; a cell that opts in resolves the
        gain / theta / distmatch / boundary steps from config. The step *names* are
        still ``ttfs_*`` (the only cell that opts in today is the cascaded cycle), but
        the RESOLUTION is now keyed by the contract, not the tuner's mode.

        ``distmatch_driven`` is whether the chosen ramp strategy (the genuine-blend
        ramp) drives the teacher-distribution matching — distmatch is not a free flag,
        it is owned by that ramp, so the pipeline records it for the tuner.
        """
        if not bool(getattr(mode_policy, "does_conversion_health_calibration", False)):
            return cls.inert()
        return cls._resolve_active(config, distmatch_driven=distmatch_driven)

    @classmethod
    def resolve(
        cls, config, *, synchronized: bool, distmatch_driven: bool,
    ) -> "CalibrationPipeline":
        """Back-compat alias: ``synchronized`` is the boolean form of the (firing ×
        sync) key. Maps to the cascaded-cycle policy (does conversion-health) when
        cascaded, else the synchronized-cycle policy (inert), then delegates to
        :meth:`for_mode`. Bit-exact with the former inline resolution."""
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

        # gain-correction: the RAMP wins over the COLD trim when both are set.
        gain_ramp = bool(get("ttfs_gain_correction_ramp", False))
        gain_cold = bool(get("ttfs_gain_correction", False)) and not gain_ramp

        # theta-cotrain is compatible_with everything except the gain RAMP.
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
            distmatch_bias_iters=int(get("ttfs_distmatch_bias_iters", 15)),
            distmatch_bias_eta=float(get("ttfs_distmatch_bias_eta", 0.7)),
        )

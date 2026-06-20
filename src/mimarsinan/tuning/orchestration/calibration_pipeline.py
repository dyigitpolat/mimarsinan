"""Composable conversion-health calibration for the TTFS-cycle tuner.

The four TTFS *calibration* concerns (numerical conversion health, largely
orthogonal to the ramp strategy) used to be resolved inline in ``_configure`` with
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
resolver: config (+ whether the schedule is synchronized, + whether the genuine-blend
ramp drives distmatch) in, a validated frozen plan of which steps are active out, with
each step's ``compatible_with`` stated and tested in one place. The tuner binds the
plan to its ``self._*`` fields; the parity-critical *application* order stays in the
tuner (gain trim before node build, theta promotion after every scalar-theta
calibration).

Compatibility rules (settled here):

* every calibration step is cascaded-only — inert when ``synchronized``;
* gain-correction has two MUTUALLY EXCLUSIVE modes, the RAMP winning over the COLD
  trim when both flags are set;
* theta-cotrain is ``compatible_with`` everything EXCEPT the gain RAMP (both manage
  ``activation_scale``; the per-depth scalar ramp would clobber the per-channel param,
  so the gain ramp wins and theta-cotrain is disabled);
* distmatch is active iff the genuine-blend ramp drives it (the ramp owns the call).
"""

from __future__ import annotations

from dataclasses import dataclass


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
    def resolve(
        cls, config, *, synchronized: bool, distmatch_driven: bool,
    ) -> "CalibrationPipeline":
        """Resolve the active calibration steps and their compatibility.

        ``distmatch_driven`` is whether the chosen ramp strategy (the genuine-blend
        ramp) drives the teacher-distribution matching — distmatch is not a free flag,
        it is owned by that ramp, so the pipeline records it for the tuner.
        """
        get = config.get

        # Every calibration step is cascaded-only.
        cascaded = not synchronized

        # gain-correction: the RAMP wins over the COLD trim when both are set.
        gain_ramp = cascaded and bool(get("ttfs_gain_correction_ramp", False))
        gain_cold = (
            cascaded
            and bool(get("ttfs_gain_correction", False))
            and not gain_ramp
        )

        # theta-cotrain is compatible_with everything except the gain RAMP.
        theta_cotrain = (
            cascaded
            and bool(get("ttfs_theta_cotrain", False))
            and not gain_ramp
        )

        boundary_ste = cascaded and bool(get("ttfs_boundary_surrogate", False))
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

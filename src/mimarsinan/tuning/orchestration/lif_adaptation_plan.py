"""Contract-driven resolution of the LIF adaptation flag-thicket."""

from __future__ import annotations

from dataclasses import dataclass

from mimarsinan.chip_simulation.spiking_semantics import is_lif
from mimarsinan.common.workload_profile import ResolvedWorkloadProfile
from mimarsinan.tuning.orchestration.lif_exact_qat import lif_exact_qat_active
from mimarsinan.tuning.orchestration.mbh_tanneal import TAnnealSchedule
from mimarsinan.tuning.orchestration.tuning_policy import FAST_LADDER_STEPS_PER_RATE

_DEFAULT_BLEND_FAST_RATES = [0.25, 0.5, 0.75, 1.0]
_GENERIC_DISTMATCH_BIAS_ITERS = 10
_GENERIC_DISTMATCH_CAL_BATCHES = 8


@dataclass(frozen=True)
class LifAdaptationPlan:
    """Every knob the LIF tuner consumes, resolved once (the TTFS-plan pattern)."""

    cycle_accurate: bool
    blend_fast_rates: list
    blend_fast_steps_per_rate: int
    blend_fast_lr_eta_min: float
    tanneal: bool
    endpoint_recovery_steps: int
    distmatch: bool
    distmatch_bias_iters: int
    distmatch_bias_eta: float
    distmatch_cal_batches: int
    theta_cotrain: bool
    simulation_steps: int
    exact_qat: bool = False

    @classmethod
    def resolve(cls, config) -> "LifAdaptationPlan":
        get = config.get
        exact_qat = lif_exact_qat_active(config)
        if exact_qat:
            # [lif_exact_qat_program §6.1(3)] the AQ stage owns the QAT: the
            # adaptation reduces to finalize+verify (rebuild LIF activations —
            # deployment-identical by A2 — and install the chip-aligned forward);
            # the T-anneal ladder, distmatch, theta-cotrain, and the raw-cascade
            # endpoint are superseded.
            return cls(
                cycle_accurate=bool(get("cycle_accurate_lif_forward", False)),
                blend_fast_rates=[1.0],
                blend_fast_steps_per_rate=0,
                blend_fast_lr_eta_min=float(get("lif_blend_fast_lr_eta_min", 0.1)),
                tanneal=False,
                endpoint_recovery_steps=0,
                distmatch=False,
                distmatch_bias_iters=0,
                distmatch_bias_eta=0.0,
                distmatch_cal_batches=0,
                theta_cotrain=False,
                simulation_steps=int(config["simulation_steps"]),
                exact_qat=True,
            )
        calibration = ResolvedWorkloadProfile.from_config(config).calibration
        bias_iters_default = (
            _GENERIC_DISTMATCH_BIAS_ITERS
            if calibration.distmatch_bias_iters is None
            else int(calibration.distmatch_bias_iters)
        )
        cal_batches_default = (
            _GENERIC_DISTMATCH_CAL_BATCHES
            if calibration.distmatch_cal_batches is None
            else int(calibration.distmatch_cal_batches)
        )
        return cls(
            cycle_accurate=bool(get("cycle_accurate_lif_forward", False)),
            blend_fast_rates=[
                float(r) for r in get("lif_blend_fast_rates", _DEFAULT_BLEND_FAST_RATES)
            ],
            blend_fast_steps_per_rate=int(
                get("lif_blend_fast_steps_per_rate", FAST_LADDER_STEPS_PER_RATE)
            ),
            blend_fast_lr_eta_min=float(get("lif_blend_fast_lr_eta_min", 0.1)),
            tanneal=(
                bool(get("lif_tanneal", False))
                and is_lif(get("spiking_mode", "lif"))
            ),
            endpoint_recovery_steps=int(get("endpoint_recovery_steps", 0)),
            distmatch=bool(get("lif_distmatch", False)),
            distmatch_bias_iters=int(get("lif_distmatch_bias_iters", bias_iters_default)),
            distmatch_bias_eta=float(get("lif_distmatch_bias_eta", 0.5)),
            distmatch_cal_batches=int(get("lif_distmatch_cal_batches", cal_batches_default)),
            theta_cotrain=bool(get("lif_theta_cotrain", False)),
            simulation_steps=int(config["simulation_steps"]),
        )

    def tanneal_schedule(self, ladder_rates) -> TAnnealSchedule | None:
        """The recipe's realizable T-anneal over the tuner's normalized ladder
        (equal budget); ``None`` keeps the value-blend recipe bit-identical."""
        if not self.tanneal:
            return None
        return TAnnealSchedule(
            target_T=self.simulation_steps,
            ladder_rates=tuple(float(r) for r in ladder_rates),
        )

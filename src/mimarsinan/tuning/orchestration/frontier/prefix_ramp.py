"""The P4 prefix-frontier strategy: converted-prefix k walked as the ramp."""

from __future__ import annotations

import torch

from mimarsinan.models.spiking.training.prefix_genuine_forward import (
    PrefixGenuineForward,
)
from mimarsinan.spiking.dfq_bias_correction import teacher_channel_means
from mimarsinan.spiking.distribution_matching import node_values_by_perceptron_index
from mimarsinan.tuning.axes.blend_axis import PrefixConversionAxis
from mimarsinan.tuning.orchestration.blend_ramp import PlainClassificationLoss
from mimarsinan.tuning.orchestration.frontier.reaffine import frontier_reaffine
from mimarsinan.tuning.orchestration.ramp_strategy import GenuineRampBase
from mimarsinan.tuning.orchestration.tuning_policy import TUNING_POLICY


class PrefixConversionRamp(GenuineRampBase):
    """P4 for multi-segment cascaded vehicles: the axis walks converted-prefix k
    (``PrefixGenuineForward``), every rung a genuine partial deployment, trained
    with plain CE (the float suffix hands the frontier segment its gradient — no
    KD teacher); the D-hat gate reads the k-hybrid, the P1'' endpoint closes at
    k=n."""

    def make_axis(self, tuner):
        return PrefixConversionAxis()

    def ramp_forward(self, tuner, model):
        tuner._prefix_forward = PrefixGenuineForward(
            model, tuner._T, rate=0.0,
            boundary_surrogate_temp=tuner._boundary_surrogate_temp,
            hop_frontier=bool(getattr(tuner, "_hop_prefix_levels", None)),
        )
        return tuner._prefix_forward

    def make_kd_loss(self, tuner):
        return PlainClassificationLoss()

    def after_install_blend_pre(self, tuner) -> None:
        super().after_install_blend_pre(tuner)
        tuner._calibrate_to_teacher_distribution()

    def on_remove_forward(self, tuner) -> None:
        tuner._prefix_forward = None


def _prefix_ann_channel_means(tuner) -> tuple:
    """(teacher per-perceptron channel means, calibration batch), cached per run."""
    cached = getattr(tuner, "_prefix_ann_mean_cache", None)
    if cached is None:
        cal_x = tuner._calibration_inputs()
        cached = (teacher_channel_means(tuner._teacher, cal_x), cal_x)
        tuner._prefix_ann_mean_cache = cached
    return cached


def run_prefix_stage_reaffine(tuner, rate: float) -> dict:
    """One P4 stage's keep-best DFQ re-affine measured through the k-hybrid.

    Sets the frontier to ``rate`` first so both the cascade means and the
    keep-best probe read the genuine partial deployment the rung will train.
    """
    tuner._fast_set_rate(float(rate))
    forward = tuner._prefix_forward
    assert forward is not None, (
        "run_prefix_stage_reaffine requires the installed PrefixGenuineForward"
    )
    ann_mean, cal_x = _prefix_ann_channel_means(tuner)

    def hybrid_channel_values() -> dict:
        with torch.no_grad():
            _, node_values = forward.forward_with_node_values(cal_x)
        return node_values_by_perceptron_index(tuner.model, node_values)

    stats = frontier_reaffine(
        tuner, ann_mean, hybrid_channel_values,
        bias_iters=int(tuner.pipeline.config.get(
            "ttfs_prefix_stage_dfq_iters", TUNING_POLICY.prefix_stage_dfq_iters,
        )),
        eta=tuner._calibration.distmatch_bias_eta,
    )
    print(
        f"[MBH-PREFIX] tuner={type(tuner).__name__} "
        f"k={forward.frontier_k}/{forward.frontier_units} rate={float(rate):.6f} "
        f"dfq_probe_entry={float(stats.get('probe_entry') or 0.0):.6f} "
        f"dfq_probe_best={float(stats.get('probe_best') or 0.0):.6f} "
        f"dfq_iters={int(stats.get('probe_iters_run', 0))}",
        flush=True,
    )
    return stats

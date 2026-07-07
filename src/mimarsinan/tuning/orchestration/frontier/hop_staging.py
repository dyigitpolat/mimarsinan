"""[5v B1(iii)] hop-staged sync AQ install: the P4 frontier generalized below segments."""

from __future__ import annotations

from mimarsinan.spiking.dfq_bias_correction import (
    teacher_activation_samples,
    teacher_channel_means,
)
from mimarsinan.spiking.gain_correction import per_perceptron_cascade_depth
from mimarsinan.tuning.orchestration.adaptation_manager import (
    HOP_DEPTH_ATTR,
    hop_frontier,
    sync_exact_qat_active,
)
from mimarsinan.tuning.orchestration.frontier.reaffine import frontier_reaffine
from mimarsinan.tuning.orchestration.install_capture import collect_channel_stats
from mimarsinan.tuning.orchestration.install_resolution import (
    build_value_install_gauge,
    value_grid_levels,
)
from mimarsinan.tuning.orchestration.mbh_ledger import (
    _fixed_validation_batch,
    _measurement_guard,
)
from mimarsinan.tuning.orchestration.tuning_policy import TUNING_POLICY

HOP_STAGE_MIN_LEVELS = 6
"""Stage only chains deeper than the measured full-recovery depth (L <= 4-5:
t0_22/t0_18/t0_03 all climbed out of equally-deep entry craters)."""

_REAFFINE_ETA = 0.7
"""DFQ step size at the frontier (the distmatch default the prefix stages use)."""


def stamp_hop_depths(model) -> int:
    """Stamp each perceptron's cascade-hop depth (survives clone deepcopies);
    returns the number of hop-depth levels."""
    depths = per_perceptron_cascade_depth(model.get_mapper_repr())
    max_depth = 0
    for perceptron in model.get_perceptrons():
        depth = int(depths.get(id(perceptron), 0))
        setattr(perceptron, HOP_DEPTH_ATTR, depth)
        max_depth = max(max_depth, depth)
    return max_depth + 1


def _install_gauge_fails(tuner, levels: int) -> bool:
    """A6(i) at the install anchor: measure the LIVE pre-install model
    (cursor-isolated) against the target grid."""
    prev_cursor = getattr(tuner.trainer, "_gpu_val_cursor", None)
    with _measurement_guard(tuner.trainer):
        batches = [x for x, _ in tuner.trainer.iter_validation_batches(2)]
        stats = collect_channel_stats(
            tuner.model, batches, tuner.pipeline.config["device"],
        )
    tuner.trainer._gpu_val_cursor = 0 if prev_cursor is None else prev_cursor
    depths = {
        id(p): int(getattr(p, HOP_DEPTH_ATTR, 0)) for p, _ in stats
    }
    thetas = [float(p.activation_scale) for p, _ in stats]
    gauge = build_value_install_gauge(stats, thetas, depths, levels)
    return gauge.fails


def resolve_sync_hop_staging(tuner):
    """Arm the hop frontier for this AQ run, or None for the monolithic install.

    Arms only when ALL hold: the ``sync_hop_staged_install`` recipe knob, sync
    exact-QAT mode, a chain deeper than the proven-recovery depth, and an A6
    install gauge that FAILS at the target grid (A6: "when A6 fails, the
    install must not be committed monolithically").
    """
    config = tuner.pipeline.config
    if not bool(config.get("sync_hop_staged_install", False)):
        return None
    if not sync_exact_qat_active(config):
        return None
    n_levels = stamp_hop_depths(tuner.model)
    if n_levels < HOP_STAGE_MIN_LEVELS:
        return None
    levels = value_grid_levels("ttfs_cycle_based", config)
    if levels is None or not _install_gauge_fails(tuner, levels):
        return None
    print(
        f"[MBH-HOP] staged sync AQ install armed: hop_levels={n_levels} "
        f"grid={levels} (A6 gauge FAIL on a chain past the recovery depth)",
        flush=True,
    )
    return n_levels


def capture_hop_reference(tuner) -> None:
    """Snapshot the pre-install reference the frontier re-affine matches:
    one fixed validation batch + the float model's per-channel means."""
    batch = _fixed_validation_batch(tuner.trainer)
    if batch is None:
        tuner._hop_cal_x = None
        tuner._hop_ann_mean = None
        return
    x, _ = batch
    tuner._hop_cal_x = x.to(tuner.pipeline.config["device"])
    with _measurement_guard(tuner.trainer):
        tuner._hop_ann_mean = teacher_channel_means(tuner.model, tuner._hop_cal_x)


def run_hop_stage_reaffine(tuner, rate):
    """Keep-best DFQ bias re-affine measured through the LIVE hop-staged install
    (T4 arm-B / W-CAL-3 at hop granularity). Returns the DFQ stats, or None
    when no reference batch exists."""
    cal_x = getattr(tuner, "_hop_cal_x", None)
    ann_mean = getattr(tuner, "_hop_ann_mean", None)
    if cal_x is None or ann_mean is None:
        return None

    def cascade_means():
        with _measurement_guard(tuner.trainer):
            return teacher_activation_samples(tuner.model, cal_x)

    stats = frontier_reaffine(
        tuner, ann_mean, cascade_means,
        bias_iters=TUNING_POLICY.prefix_stage_dfq_iters,
        eta=_REAFFINE_ETA,
    )
    k = hop_frontier(float(rate), int(tuner._hop_stage_levels or 0))
    print(
        f"[MBH-HOP] stage reaffine rate={float(rate):.4f} frontier_k={k} "
        f"probe_entry={stats.get('probe_entry')} "
        f"probe_best={stats.get('probe_best')} "
        f"iters={stats.get('probe_iters_run')}",
        flush=True,
    )
    return stats

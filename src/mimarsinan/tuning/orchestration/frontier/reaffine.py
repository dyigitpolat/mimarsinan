"""The frontier rung repair: keep-best DFQ bias re-affine at the live frontier."""

from __future__ import annotations

from mimarsinan.spiking.dfq_bias_correction import dfq_correct_biases
from mimarsinan.tuning.orchestration.mbh_ledger import live_model_acc_fp32
from mimarsinan.tuning.orchestration.tuning_policy import TUNING_POLICY


def frontier_reaffine(tuner, ann_mean, cascade_means, *, bias_iters, eta) -> dict:
    """One rung's keep-best DFQ bias re-affine toward ``ann_mean`` (T4 arm-B /
    W-CAL-3 at frontier granularity), probed through the LIVE partial
    deployment — ratcheted, so the rung can never end below its entry state.
    """
    return dfq_correct_biases(
        tuner.model,
        ann_mean,
        cascade_means,
        bias_iters=int(bias_iters),
        eta=float(eta),
        probe=lambda: live_model_acc_fp32(tuner),
        probe_patience=TUNING_POLICY.dfq_keepbest_patience,
    )

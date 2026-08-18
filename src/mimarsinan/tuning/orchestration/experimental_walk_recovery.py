"""EXPERIMENTAL, default-off: exact-QAT endpoint recovery through the deployed walk.

Everything the ``lif_exact_qat_walk_recovery`` knob does lives HERE — the
mainline tuners each make one call into this module and are bitwise
incumbent when the knob is off (proven: fresh t0_01 0.9799 / t0_30 0.9587
reproduce the incumbent to the digit).

What arming does: (1) the AQ endpoint recovery — the freshest, largest
ledger leg, rollback-guarded — grinds the value-domain chip-aligned walk
instead of the plain flow it is otherwise blind to; (2) the LIF Adaptation
endpoint recovery keeps the recipe budget instead of the exact-QAT
zero-step reduction. Both engage only on graphs with host ComputeOps.

When it may be useful: configs where the incumbent curriculum COLLAPSES —
short-T offload/multi-segment graphs whose install cliff dwarfs the WQ
recovery (measured +10.3pp on the offloaded mixer at T=4: 0.8144 vs
0.7111 fresh, all certificates green).

When NOT to arm (measured, memo §13): healthy tier cells — knob-ON reads
−0.31pp (t0_01), −1.34pp (t0_30), flat elsewhere, and one nevresim FATAL
certificate failure (t0_05: the shifted trajectory trained onto boundary
knife-edges). The default stays off until the MBH ledger split is retuned
for the honest objective.
"""

from __future__ import annotations

from mimarsinan.tuning.orchestration import adaptation_ledger
from mimarsinan.tuning.orchestration.lif_exact_qat import lif_exact_qat_active


def walk_recovery_armed(pipeline_config) -> bool:
    return bool(pipeline_config.get("lif_exact_qat_walk_recovery", False)) and (
        lif_exact_qat_active(pipeline_config)
    )


def exact_qat_training_forward(model, pipeline_config):
    """The value-domain chip-aligned walk as a ``model.forward`` override
    (staircase hops are theorem-equal to LIF hops, calculus §16)."""
    from mimarsinan.tuning.forward_install import ChipAlignedNFForward

    return ChipAlignedNFForward(
        model, int(pipeline_config["simulation_steps"]), synchronized=True,
    )


def install_walk_for_aq_recovery(tuner) -> bool:
    """Arm the AQ endpoint recovery with the deployed walk (host-op graphs
    only); the installed forward persists until the LIF finalize handoff."""
    if not walk_recovery_armed(tuner.pipeline.config):
        return False
    # Lazy: the spiking package init pulls chip_simulation (house cycle).
    from mimarsinan.spiking.segment_partition import graph_has_host_compute_ops

    if not graph_has_host_compute_ops(tuner.model):
        return False
    tuner._install_forward(
        exact_qat_training_forward(tuner.model, tuner.pipeline.config)
    )
    adaptation_ledger.record_escalation(
        adaptation_ledger.ledger_of(tuner),
        adaptation_ledger.WALK_RECOVERY_INSTALL,
        "AQ endpoint recovery grinds the deployed chip-aligned walk",
    )
    return True


def restore_lif_recovery_budget(plan, tuner):
    """The D3 leg: keep the recipe's LIF endpoint-recovery budget on host-op
    graphs instead of the exact-QAT zero-step reduction."""
    if not plan.exact_qat or not walk_recovery_armed(tuner.pipeline.config):
        return plan
    from mimarsinan.spiking.segment_partition import graph_has_host_compute_ops

    if not graph_has_host_compute_ops(tuner.model):
        return plan
    adaptation_ledger.record_escalation(
        adaptation_ledger.ledger_of(tuner),
        adaptation_ledger.LIF_RECOVERY_BUDGET_RESTORE,
        "recipe LIF endpoint-recovery budget kept over the exact-QAT reduction",
    )
    return plan.restore_recovery_for_host_graph(tuner.pipeline.config)

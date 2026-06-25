"""D4 structured pre-mapping pruning hook for :class:`SoftCoreMappingStep`.

Applies :func:`mimarsinan.transformations.pruning.magnitude.prune_perceptron_chain`
to the fused model BEFORE it is mapped to the IR, keyed on the opt-in
``prune_sparsity`` deployment parameter (resolved on the :class:`DeploymentPlan`).
This structurally removes the lowest-magnitude OUTPUT channels of each perceptron
and propagates the removal into the next layer's INPUT columns, so the mapped
softcore shapes shrink and the deployment maps to FEWER hard cores / reprogram
phases (the D4 cost lever the A2 pruning screen consumes).

Runs at SCM time, i.e. AFTER ``NormalizationFusionStep`` has folded each
perceptron's normalization into its ``nn.Linear`` (``normalization == Identity``),
so shrinking the linear's output rows is structurally sound — there is no stale
affine-norm vector left to mismatch the pruned width.

DEFAULT-OFF / BYTE-IDENTICAL: ``prune_sparsity`` unset or ``0.0`` ⇒ this is a no-op
that returns ``None`` and leaves every ``nn.Linear`` object (and its parameter
tensors) untouched, so the default deployment path is provably unchanged.
"""

from __future__ import annotations

from mimarsinan.common.diagnostics import phase_profiler
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan


def apply_structured_pruning_if_enabled(step, model, phase_tag: str):
    """Structurally prune ``model`` before mapping when ``prune_sparsity > 0``.

    Returns the :class:`ChannelPruningResult` when pruning ran, else ``None`` (the
    byte-identical default). Mutates each perceptron's ``.layer`` in place, so the
    model's cached mapper representation (which holds the same perceptron objects)
    picks up the smaller layers without rebuilding.
    """
    sparsity = float(DeploymentPlan.of(step.pipeline).prune_sparsity)
    if sparsity <= 0.0:
        return None

    from mimarsinan.transformations.pruning.magnitude import prune_perceptron_chain

    perceptrons = model.get_perceptrons()
    counts_before = [int(p.layer.out_features) for p in perceptrons]
    with phase_profiler(phase_tag, "structured_pruning"):
        result = prune_perceptron_chain(perceptrons, sparsity)
    counts_after = [int(p.layer.out_features) for p in perceptrons]
    print(
        f"[SoftCoreMappingStep] Structured pre-mapping pruning "
        f"(prune_sparsity={sparsity}): output channels "
        f"{counts_before} -> {counts_after}"
    )
    return result

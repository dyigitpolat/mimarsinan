"""D4 structured pre-mapping magnitude pruning hook for :class:`SoftCoreMappingStep`."""

from __future__ import annotations

from mimarsinan.common.diagnostics import phase_profiler
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.transformations.pruning.magnitude import prune_perceptron_chain


def apply_structured_pruning_if_enabled(step, model, phase_tag: str):
    """Structurally prune ``model`` before mapping when ``prune_sparsity > 0``.

    Returns the ``ChannelPruningResult`` when pruning ran, else ``None`` (the
    byte-identical default); mutates each perceptron's ``.layer`` in place.
    """
    sparsity = float(DeploymentPlan.of(step.pipeline).prune_sparsity)
    if sparsity <= 0.0:
        return None

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

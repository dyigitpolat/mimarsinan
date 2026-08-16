"""D4 structured pre-mapping pruning hook for :class:`SoftCoreMappingStep`,
with the W3b ``prune_criterion`` selection seam (criterion agnosticism).

``prune_criterion`` (registry default ``row_col_l1``) picks how the one-shot
pruning at soft-core mapping time expresses its decision:

- ``row_col_l1`` (default, byte-identical with the key absent): the incumbent
  structural channel shrink (``magnitude.prune_perceptron_chain``).
- ``activation`` / ``partial_column_group``: FOREIGN seed criteria — instead
  of shrinking layers, install committed seed masks
  (``transformations.pruning.seed_generators``) that the IR pruning cascade
  consumes through ``get_initial_pruning_masks_from_model`` and harvests.
"""

from __future__ import annotations

from mimarsinan.common.diagnostics import phase_profiler
from mimarsinan.common.workload_profile import ResolvedWorkloadProfile
from mimarsinan.config_schema.registry import effective_value as _effective
from mimarsinan.mapping.pruning.boundary_policy import (
    build_boundary_ir_graph,
    compute_perceptron_io_exemption_indices,
)
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.registry.trainer_factory import make_basic_trainer
from mimarsinan.transformations.pruning import collect_activation_stats
from mimarsinan.transformations.pruning.magnitude import prune_perceptron_chain
from mimarsinan.transformations.pruning.seed_generators import (
    DEFAULT_PARTIAL_COLUMN_GROUP_SIZE,
    DEFAULT_PRUNE_CRITERION,
    SeedContext,
    generate_seed_masks,
    install_seed_masks,
)


def resolve_prune_criterion(config) -> str:
    """The effective one-shot pruning criterion (registry default row_col_l1)."""
    return str(_effective(config, "prune_criterion") or DEFAULT_PRUNE_CRITERION)


def _collect_seed_activation_stats(step, model):
    """Collect activation-importance stats for the 'activation' seed criterion
    (same instrument the pruning tuner uses: ``collect_activation_stats`` over
    validation batches, count from the workload calibration profile)."""
    trainer = make_basic_trainer(step.pipeline, model)
    declared = ResolvedWorkloadProfile.from_config(
        step.pipeline.config
    ).calibration.stat_batches
    return collect_activation_stats(
        model,
        trainer.validation_loader,
        step.pipeline.config.get("device", "cpu"),
        num_batches=5 if declared is None else int(declared),
    )


def _seed_context_for(step, model, criterion: str) -> SeedContext:
    """Build the criterion inputs: IR-derived boundary exemptions plus the
    criterion-specific extras (activation stats / group size)."""
    exempt_in, exempt_out = compute_perceptron_io_exemption_indices(
        build_boundary_ir_graph(
            model,
            weight_bits=int(step.pipeline.config.get("weight_bits", 8)),
            firing_mode=str(step.pipeline.config.get("firing_mode", "Default")),
        ),
        model.get_perceptrons(),
    )
    activation_stats = (
        _collect_seed_activation_stats(step, model)
        if criterion == "activation" else None
    )
    group_size = int(
        _effective(step.pipeline.config, "prune_group_size")
        or DEFAULT_PARTIAL_COLUMN_GROUP_SIZE
    )
    return SeedContext(
        activation_stats=activation_stats,
        group_size=group_size,
        exempt_input_layers=frozenset(exempt_in),
        exempt_output_layers=frozenset(exempt_out),
    )


def apply_structured_pruning_if_enabled(step, model, phase_tag: str):
    """Prune ``model`` before mapping when ``prune_sparsity > 0``.

    Default criterion (``row_col_l1``, byte-identical with ``prune_criterion``
    absent): structural channel shrink; returns the ``ChannelPruningResult``.
    Foreign criteria: install committed seed masks for the IR cascade (layers
    keep their shapes); returns ``None``. ``prune_sparsity <= 0`` is always
    the byte-identical no-op.
    """
    sparsity = float(DeploymentPlan.of(step.pipeline).prune_sparsity)
    if sparsity <= 0.0:
        return None

    criterion = resolve_prune_criterion(step.pipeline.config)
    if criterion == DEFAULT_PRUNE_CRITERION:
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

    perceptrons = model.get_perceptrons()
    with phase_profiler(phase_tag, "seed_mask_pruning"):
        context = _seed_context_for(step, model, criterion)
        seeds = generate_seed_masks(criterion, perceptrons, sparsity, context)
        install_seed_masks(perceptrons, seeds)
    pruned_fractions = [
        float(s.element_pruned.float().mean().item()) for s in seeds
    ]
    completed = [
        (int(s.row_pruned.sum().item()), int(s.col_pruned.sum().item()))
        for s in seeds
    ]
    print(
        f"[SoftCoreMappingStep] Seed-mask pre-mapping pruning "
        f"(prune_criterion={criterion!r}, prune_sparsity={sparsity}): "
        f"element fractions {pruned_fractions}, completed (rows, cols) "
        f"{completed}"
    )
    return None

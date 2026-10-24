"""The shape-only specs of one segment's IR cores, the pass planner's input on the builder's plane (``_segment_specs``,
at the declared pool's weight-sign granularity through ``segment_specs_for``)."""

from __future__ import annotations

import functools
from typing import Sequence

from mimarsinan.mapping.ir import IRGraph, NeuralCore
from mimarsinan.mapping.platform.platform_constraints import PER_AXON, PER_SYNAPSE, pool_sign_granularity


_SPLIT_FALLBACK_LOGGED = False


def segment_specs_for(cores_config: Sequence[dict]):
    """``_segment_specs`` at the declared pool's weight-sign granularity (the planner's spec reader)."""
    return functools.partial(_segment_specs, sign_granularity=pool_sign_granularity(cores_config))


def _segment_specs(
    cores: list[NeuralCore],
    *,
    ir_graph: IRGraph | None = None,
    hardware_bias: bool = False,
    sign_granularity: str = PER_SYNAPSE,
) -> "tuple[list, dict[int, NeuralCore], list[int | None]]":
    """Shape-only specs for one segment's IR cores, sized POST-compaction (and, under per-axon sign, at the
    exact physical rows the runtime soft cores pack: ``spec_at_deployed_extent``).

    [U1] The pass PLANNING moved to ``pass_planner`` (one planner, both
    planes); this only produces its inputs: the specs, the spec→core map the
    materialization walks back through, and the coalescing group ids that
    keep a wide fan-in's fragments in one pass. Every spec is sized by the
    post-compaction extent: elimination reaches the pass ladder only if a
    pruned instance is budgeted for what it occupies.
    """
    global _SPLIT_FALLBACK_LOGGED
    if not cores:
        return [], {}, []

    from dataclasses import replace

    from mimarsinan.mapping.layout.softcore_spec_adapter import (
        spec_at_compacted_extent,
        spec_at_deployed_extent,
        spec_from_neural_core,
    )
    from mimarsinan.mapping.platform.core_residency import (
        BASIS_PROVENANCE, provenance_group_id, ungrouped_fallback_id)

    layout_specs = list(getattr(ir_graph, "layout_softcores", None) or [])
    use_layout = bool(layout_specs) and all(
        getattr(c, "layout_softcore_index", None) is not None for c in cores
    )

    specs: list = []
    coalescing_group_ids: list[int | None] = []
    spec_to_core: dict[int, NeuralCore] = {}
    for idx, core in enumerate(cores):
        if use_layout:
            sc_idx = int(core.layout_softcore_index)  # type: ignore[arg-type]
            spec = spec_at_compacted_extent(layout_specs[sc_idx], core)
            # [W04] The class the packer reads on this core's soft core (a segment subgraph carries no layout record:
            # the source perceptron's), as the shape-only walk keys it; the record's values class merged layers the
            # packer keeps apart, and the build refused its own plan (``PlannedPassPackingRefusal``).
            spec = replace(spec, residency_basis=BASIS_PROVENANCE, residency_class_id=provenance_group_id(
                getattr(core, "perceptron_index", None), fallback=ungrouped_fallback_id(sc_idx)))
        else:
            if not _SPLIT_FALLBACK_LOGGED:
                import logging

                logging.getLogger(__name__).warning(
                    "Scheduled split: reconstructing LayoutSoftCoreSpec from NeuralCore "
                    "(missing IRGraph.layout_softcores); input_count may diverge from SCM, "
                    "and residency classes fall back to the source-perceptron proxy "
                    "(residency_basis=provenance_fallback) rather than the core values."
                )
                _SPLIT_FALLBACK_LOGGED = True
            spec = spec_from_neural_core(
                core,
                hardware_bias=hardware_bias,
                fallback_residency_class_id=ungrouped_fallback_id(idx),
            )
        spec = spec_at_deployed_extent(spec, core, sign_granularity, graph=ir_graph)
        bank_id = None if sign_granularity == PER_AXON else getattr(core, "weight_bank_id", None)
        if spec.bank_id is None and bank_id is not None:
            # The core is the authority on bank identity: a layout spec that
            # predates bank stamping must not silently disqualify residency.
            spec = replace(spec, bank_id=int(bank_id))
        specs.append(spec)
        coalescing_group_ids.append(getattr(core, "coalescing_group_id", None))
        spec_to_core[id(spec)] = core

    return specs, spec_to_core, coalescing_group_ids

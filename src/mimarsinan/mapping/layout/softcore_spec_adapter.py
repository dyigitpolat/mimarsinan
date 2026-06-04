"""Adapters that derive a shape-only ``LayoutSoftCoreSpec`` from concrete cores.

These let the placement engine build a layout plan from cores that already
exist in a different representation:

- :func:`spec_from_neural_core` -- from an IR ``NeuralCore`` (pre-flush, used by
  scheduled capacity splitting).
- :func:`spec_from_softcore` -- from a compacted runtime ``SoftCore`` (post
  pruning compaction), so the deployment placement plan is derived from the
  exact shapes the runtime packer will see.
"""

from __future__ import annotations

from typing import Any

from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec


def spec_from_neural_core(
    core: Any,
    *,
    hardware_bias: bool,
    fallback_threshold_group_id: int,
) -> LayoutSoftCoreSpec:
    """Reconstruct a ``LayoutSoftCoreSpec`` from an IR ``NeuralCore``.

    ``fallback_threshold_group_id`` is used when the core has no
    ``perceptron_index`` (synthesised accumulator cores, etc.).
    """
    from mimarsinan.mapping.platform.mapping_structure import compute_core_input_count

    lat = int(core.latency) if core.latency is not None else 0
    pi = getattr(core, "perceptron_index", None)
    tg = int(pi) if pi is not None else int(fallback_threshold_group_id)

    n_sources = int(len(core.input_sources.flatten()))
    has_bias_axon = core.hardware_bias is None and any(
        getattr(s, "is_always_on", lambda: False)()
        for s in core.input_sources.flatten()
    )
    in_count = compute_core_input_count(
        n_sources - (1 if has_bias_axon else 0),
        has_bias=has_bias_axon,
        hardware_bias=hardware_bias,
    )
    return LayoutSoftCoreSpec(
        input_count=in_count,
        output_count=int(core.get_output_count()),
        threshold_group_id=tg,
        latency_tag=lat,
        segment_id=0,
        name=core.name,
    )


def spec_from_softcore(
    softcore: Any,
    *,
    fallback_threshold_group_id: int,
) -> LayoutSoftCoreSpec:
    """Reconstruct a ``LayoutSoftCoreSpec`` from a (compacted) runtime ``SoftCore``.

    The runtime core already carries its physical axon/neuron counts, so the
    spec mirrors exactly what the runtime packer will pack -- including any
    pruning compaction already applied.
    """
    pi = getattr(softcore, "perceptron_index", None)
    tg = int(pi) if pi is not None else int(fallback_threshold_group_id)
    lat = (
        int(softcore.latency)
        if getattr(softcore, "latency", None) is not None
        else 0
    )
    return LayoutSoftCoreSpec(
        input_count=int(softcore.get_input_count()),
        output_count=int(softcore.get_output_count()),
        threshold_group_id=tg,
        latency_tag=lat,
        segment_id=0,
        name=getattr(softcore, "name", None),
    )

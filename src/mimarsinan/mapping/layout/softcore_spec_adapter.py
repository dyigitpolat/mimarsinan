"""Adapters deriving a shape-only ``LayoutSoftCoreSpec`` from an IR ``NeuralCore`` or a runtime ``SoftCore``."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec
from mimarsinan.mapping.packing.softcore import (
    compacted_core_extent,
    eliminated_core_extent,
)
from mimarsinan.mapping.platform.mapping_structure import compute_core_input_count
from mimarsinan.mapping.platform.core_residency import (
    BASIS_PROVENANCE,
    provenance_group_id,
)


def spec_from_neural_core(
    core: Any,
    *,
    hardware_bias: bool,
    fallback_residency_class_id: int,
) -> LayoutSoftCoreSpec:
    """Reconstruct a ``LayoutSoftCoreSpec`` from an IR ``NeuralCore``, sized by
    the axon/neuron counts the runtime packer will pack (post-compaction);
    ``fallback_residency_class_id`` applies when the core has no ``perceptron_index``."""
    lat = int(core.latency) if core.latency is not None else 0
    tg = provenance_group_id(
        getattr(core, "perceptron_index", None), fallback=fallback_residency_class_id
    )

    n_axons, n_neurons = compacted_core_extent(core)
    has_bias_axon = core.hardware_bias is None and any(
        getattr(s, "is_always_on", lambda: False)()
        for s in core.input_sources.flatten()
    )
    in_count = compute_core_input_count(
        n_axons - (1 if has_bias_axon else 0),
        has_bias=has_bias_axon,
        hardware_bias=hardware_bias,
    )
    return LayoutSoftCoreSpec(
        input_count=in_count,
        output_count=n_neurons,
        residency_class_id=tg,
        residency_basis=BASIS_PROVENANCE,
        latency_tag=lat,
        segment_id=0,
        name=core.name,
    )


def spec_at_compacted_extent(spec: LayoutSoftCoreSpec, core: Any) -> LayoutSoftCoreSpec:
    """Re-size a PRE-elimination layout record onto the extent ``core`` will occupy.

    A layout softcore is recorded during the shape walk, before anything is
    eliminated, so scheduling it verbatim budgets for a crossbar the packer
    will never place. An untouched geometry is returned unchanged.
    """
    eliminated_axons, eliminated_neurons = eliminated_core_extent(core)
    if not (eliminated_axons or eliminated_neurons):
        return spec
    return replace(
        spec,
        input_count=int(spec.input_count) - eliminated_axons,
        output_count=int(spec.output_count) - eliminated_neurons,
    )


def spec_from_softcore(
    softcore: Any,
    *,
    fallback_residency_class_id: int,
) -> LayoutSoftCoreSpec:
    """Reconstruct a ``LayoutSoftCoreSpec`` from a compacted runtime ``SoftCore``,
    mirroring the exact axon/neuron counts the runtime packer will pack."""
    tg = provenance_group_id(
        getattr(softcore, "perceptron_index", None), fallback=fallback_residency_class_id
    )
    lat = (
        int(softcore.latency)
        if getattr(softcore, "latency", None) is not None
        else 0
    )
    return LayoutSoftCoreSpec(
        input_count=int(softcore.get_input_count()),
        output_count=int(softcore.get_output_count()),
        residency_class_id=tg,
        residency_basis=BASIS_PROVENANCE,
        latency_tag=lat,
        segment_id=0,
        name=getattr(softcore, "name", None),
    )

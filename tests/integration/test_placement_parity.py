"""Phase 2 placement parity: the shape-only layout packer (``pack_layout``) and
the weight-bearing runtime packer (``HardCoreMapping.map``) must produce the
same hardware placement (cores used + per-core dimensions + fused dims) when fed
the *same* compacted soft cores.

This is the structural half of "layout-only mapper as the single source of
truth": the wizard miniview (layout packer) cannot disagree with deployment
(runtime packer) about how many cores are used or how wide/fused they are.
"""

from __future__ import annotations

import pytest

from integration._placement_signature import CONFIGS

from mimarsinan.mapping.layout.layout_packer import pack_layout
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType
from mimarsinan.mapping.layout.softcore_spec_adapter import spec_from_softcore
from mimarsinan.mapping.packing.neural_segment_packing import (
    neural_segment_to_soft_core_mapping,
)
from mimarsinan.mapping.packing.softcore import compact_soft_core_mapping
from mimarsinan.mapping.packing.softcore.hard_core_mapping import HardCoreMapping
from mimarsinan.mapping.packing.hybrid_segment_helpers import (
    _make_available_hardware_cores,
)


# Neural-only, single-pool configs (no host ComputeOp barrier, no scheduling):
# for these the whole graph is one neural segment, so we can pack it directly.
_SINGLE_SEGMENT = ["dense_two_core", "fusion_wide_axon", "neuron_split", "pruned"]


def _compacted_soft_mapping(ir):
    soft = neural_segment_to_soft_core_mapping(ir, {})
    compact_soft_core_mapping(soft.cores, soft.output_sources)
    return soft


def _hardcore_dims(cores):
    return sorted(
        (int(hc.axons_per_core), int(hc.neurons_per_core)) for hc in cores
    )


@pytest.mark.parametrize("name", _SINGLE_SEGMENT)
def test_layout_and_runtime_packers_agree(name):
    builder, kwargs = CONFIGS[name]
    cores_config = kwargs["cores_config"]
    allow_neuron_splitting = kwargs.get("allow_neuron_splitting", False)

    core_types = [
        LayoutHardCoreType(
            max_axons=int(ct["max_axons"]),
            max_neurons=int(ct["max_neurons"]),
            count=int(ct["count"]),
        )
        for ct in cores_config
    ]

    # Layout packer over specs derived from the compacted soft cores.
    soft_layout = _compacted_soft_mapping(builder())
    specs = [
        spec_from_softcore(c, fallback_threshold_group_id=-(i + 1))
        for i, c in enumerate(soft_layout.cores)
    ]
    layout_result = pack_layout(
        softcores=specs,
        core_types=core_types,
        allow_neuron_splitting=allow_neuron_splitting,
    )

    # Runtime packer over the same compacted soft cores.
    soft_runtime = _compacted_soft_mapping(builder())
    hcm = HardCoreMapping(_make_available_hardware_cores(cores_config))
    hcm.map(soft_runtime, allow_neuron_splitting=allow_neuron_splitting)

    assert layout_result.feasible
    assert layout_result.cores_used == len(hcm.cores), (
        f"{name}: cores_used drift layout={layout_result.cores_used} "
        f"runtime={len(hcm.cores)}"
    )
    assert _hardcore_dims(hcm.cores) == sorted(
        (s.axons_per_core, s.neurons_per_core)
        for s in (layout_result.used_core_snapshots or ())
    ), f"{name}: per-core dimension drift between layout and runtime packers"

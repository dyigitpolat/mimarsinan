"""Publishing a SANA-FE segment's output raster to a later pass of its segment."""

from __future__ import annotations

from mimarsinan.chip_simulation.sanafe.analysis import _group_name
from mimarsinan.models.spiking.hybrid.carry import (
    carried_output_ids,
    publish_carried_trains,
)


def publish_sanafe_carry(
runner, stage, state_buffer_spikes, *, seg_raster, group_row_offsets, hcm,
core_to_group,
) -> None:
    """Hand a later pass of this segment the producer's own spike train.

    Only wires a later PASS reads are published; a wire crossing to a later
    SEGMENT is a host boundary and keeps its count re-encode by design.
    """
    if state_buffer_spikes is None:
        return
    carried = carried_output_ids(runner.mapping).get(
        _stage_index_of(runner.mapping, stage))
    if not carried:
        return
    rows = {
        index: int(group_row_offsets.get(_group_name(core_to_group[index]), 0))
        for index in range(len(hcm.cores)) if index in core_to_group
    }
    latency = {
        index: int(getattr(core, "latency", 0) or 0)
        for index, core in enumerate(hcm.cores)
    }
    raster = runner._compute_seg_output_raster(
        hcm.output_sources, seg_raster=seg_raster, core_rows=rows,
        core_latency=latency, T=runner.T,
    )
    if raster is None:
        return
    publish_carried_trains(stage, raster, carried, state_buffer_spikes)


def _stage_index_of(mapping, stage) -> int:
    for index, candidate in enumerate(mapping.stages):
        if candidate is stage:
            return index
    return -1

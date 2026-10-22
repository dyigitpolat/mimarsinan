"""Publishing a SANA-FE segment's output raster to a later pass of its segment."""

from __future__ import annotations

from mimarsinan.chip_simulation.sanafe.analysis import _group_name
from mimarsinan.chip_simulation.sanafe.neuron_model import input_neuron_attributes
from mimarsinan.models.spiking.hybrid.carry import (
    carried_output_ids,
    carry_scope,
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
    publish_carried_trains(stage, raster, carried, carry_scope(state_buffer_spikes, stage, publish=True))


def _stage_index_of(mapping, stage) -> int:
    for index, candidate in enumerate(mapping.stages):
        if candidate is stage:
            return index
    return -1


def present_inputs_in_core_time(core_input_neurons, hcm, encoded) -> None:
    """[U1 R2] Every stage-input row reaches each core that reads it in that core's local time, as the HCM reads a
    stage input (``lif_step``): SANA-FE's input neurons spike at the global cycle, so the neuron feeding a core that
    starts at latency L > 0 replays its row of ``encoded`` ``(1, size, T')`` L cycles late. A carried slice and a
    host-encoded one alike: an ALAP-raised reader of a host slice read it L cycles late before, identity programs
    included."""
    for (core_idx, axon_idx), neuron in core_input_neurons.items():
        latency = int(getattr(hcm.cores[core_idx], "latency", 0) or 0)
        k = int(hcm.cores[core_idx].axon_sources[axon_idx].neuron_)
        if latency > 0 and k < encoded.shape[1]:
            train = [0] * latency + [int(b) for b in encoded[0, k, :max(0, encoded.shape[2] - latency)].tolist()]
            neuron.set_attributes(model_attributes=input_neuron_attributes(train))

"""[wsm V2] Bank-clustered residency: verification and storage dedup of the
planner's streamed composition (the composition itself lives in
``support.schedule.pass_planner`` — one planner, both planes [U1])."""

from __future__ import annotations

from mimarsinan.mapping.packing.softcore import eliminated_core_extent
from mimarsinan.mapping.platform.mapping_structure import PositionMultiplexingError
from mimarsinan.mapping.support.schedule.pass_planner import resident_passes


def _stage_geometry(stage) -> list:
    segment = stage.hard_core_mapping
    geometry = []
    for placements in segment.soft_core_placements_per_hard_core:
        geometry.append(frozenset(
            (placement.get("weight_bank_id"), int(placement["axons"]),
             int(placement["neurons"]), int(placement["axon_offset"]),
             int(placement["neuron_offset"]))
            for placement in placements
        ))
    return geometry


def dedup_resident_stage_matrices(pass_stages: list) -> None:
    """[wsm V3] Storage dedup across a bank-clustered pass chain.

    Duplicate resident cores hold bitwise-identical padded grids. Descriptor
    cores already reference ONE shared payload per bank, so they need no
    rebinding; this only aliases cores that OWN a dense grid (legacy or
    externally written mappings), within the head stage by (bank, region,
    geometry) and across passes at each ordinal whose placement geometry is
    EQUAL (the verified residency law) — so pickle memoization stores each
    distinct payload once (measured: the scheduled ViT materialized ~4.9k
    grids into 13 GB per mapping pickle).
    """
    head = pass_stages[0].hard_core_mapping
    shared: dict = {}
    for core, placements in zip(
        head.cores, head.soft_core_placements_per_hard_core
    ):
        if core.core_matrix is None or len(placements) != 1:
            continue
        record = placements[0]
        if record.get("weight_bank_id") is None:
            continue
        key = (
            record["weight_bank_id"],
            record.get("bank_axon_range"), record.get("bank_neuron_range"),
            record["axon_offset"], record["neuron_offset"],
            record["axons"], record["neurons"],
            core.core_matrix.shape, str(core.core_matrix.dtype),
        )
        existing = shared.get(key)
        if existing is None:
            shared[key] = core.core_matrix
        else:
            core.core_matrix = existing
    head_geometry = _stage_geometry(pass_stages[0])
    for stage in pass_stages[1:]:
        geometry = _stage_geometry(stage)
        for i, core in enumerate(stage.hard_core_mapping.cores):
            head_core = head.cores[i]
            if core.core_matrix is None or head_core.core_matrix is None:
                continue
            if (
                geometry[i] == head_geometry[i]
                and core.core_matrix.shape == head_core.core_matrix.shape
                and core.core_matrix.dtype == head_core.core_matrix.dtype
            ):
                core.core_matrix = head_core.core_matrix


def mark_bank_residency(pass_stages: list) -> None:
    """Verify every pass p>0 places, per physical core ordinal, a SUBSET of
    pass 0's (bank, region) placements — regions already programmed, so zero
    new programming (the ragged final pass uses fewer regions). Then mark
    those stages weight-resident. Any divergence is a scheduler bug — fail
    loud, never silently claim residency."""
    if len(pass_stages) <= 1:
        return
    reference = _stage_geometry(pass_stages[0])
    residency = resident_passes(len(pass_stages), policy_applied=True)
    for stage, resident in zip(pass_stages[1:], residency[1:]):
        geometry = _stage_geometry(stage)
        if len(geometry) > len(reference) or any(
            not geometry[i] <= reference[i] for i in range(len(geometry))
        ):
            raise RuntimeError(
                f"bank_clustered residency violated at pass "
                f"{stage.schedule_pass_index} of segment "
                f"{stage.schedule_segment_index}: placement geometry diverged "
                f"from pass 0 — weights would silently reprogram."
            )
        stage.schedule_weights_resident = resident


def stage_copies(stage) -> list:
    """Per physical core: its shape and the bank regions it holds (bank, bank ranges, extent, offsets)."""
    segment = stage.hard_core_mapping
    return [
        (int(core.axons_per_core), int(core.neurons_per_core), frozenset(
            (placement.get("weight_bank_id"), placement.get("bank_axon_range"), placement.get("bank_neuron_range"),
             int(placement["axons"]), int(placement["neurons"]), int(placement["axon_offset"]),
             int(placement["neuron_offset"]))
            for placement in placements))
        for core, placements in zip(segment.cores, segment.soft_core_placements_per_hard_core)
    ]


def stage_soft_cores(stage) -> list:
    """[thru:w5] Per physical core: the soft cores it holds, each as its IR node and offsets (a soft core's weights are
    its IR node's)."""
    return [frozenset((int(p["ir_node_id"]), int(p["axon_offset"]), int(p["neuron_offset"])) for p in placements)
            for placements in stage.hard_core_mapping.soft_core_placements_per_hard_core]


def mark_multiplexed_residency(pass_stages: list, run_heads) -> None:
    """[W02] A pass that is not its run's first executes on the copies the first programmed: verify that every
    physical core holds exactly the head's bank regions, each bank-backed, then mark the pass weight-resident; any
    divergence is refused by name, never claimed as residency. [thru:w5] A pass that serves another input slot of its
    head's own planned pass (``slot_turns``) holds the head's very soft cores, which need no bank: the same IR nodes at
    the same offsets of every core."""
    if len(pass_stages) != len(run_heads):
        raise PositionMultiplexingError(
            f"position_multiplexing: {len(pass_stages)} executed passes for {len(run_heads)} planned ones")
    head = head_cores = None
    for stage, is_head in zip(pass_stages, run_heads):
        copies = stage_copies(stage)
        if is_head:
            head, head_cores = copies, stage_soft_cores(stage)
            continue
        banked = all(region[0] is not None for _a, _n, regions in copies for region in regions)
        if copies != head or not (banked or stage_soft_cores(stage) == head_cores):
            raise PositionMultiplexingError(
                f"position_multiplexing: pass {stage.schedule_pass_index} of segment {stage.schedule_segment_index} "
                f"places other weights than the first pass of its run, so it cannot execute on that pass's copies")
        stage.schedule_weights_resident = True


def multiplexing_shares(cores: list, ir_graph, label: str, position_multiplexing) -> list:
    """[W02] Per IR core whether it shares its bank's one copy (a bank instance IR elimination left whole), for the
    pass planner; under ``position_multiplexing`` an IR without the layout walk's record, or a compacted bank instance,
    refuses by name (moved from ``hybrid_build_scheduled``). [U-L] A core the lane fusion inserted carries a record of
    its own (its level in the fused program), so a pool lane's cores pass; a core inserted without one does not."""
    if position_multiplexing and not (getattr(ir_graph, "layout_softcores", None) and all(
            getattr(c, "layout_softcore_index", None) is not None for c in cores)):
        raise PositionMultiplexingError(
            f"position_multiplexing orders {label!r}'s passes by the layout walk's dependency "
            f"levels, and this IR carries no layout record for its cores")
    shares = [c.weight_bank_id is not None and eliminated_core_extent(c) == (0, 0) for c in cores]
    if position_multiplexing and any(c.weight_bank_id is not None and not ok for c, ok in zip(cores, shares)):
        raise PositionMultiplexingError(
            f"position_multiplexing: IR elimination compacted bank instances of {label!r} into owned "
            "cores, which the shape-only plane plans as sharing their bank's copy; the planes would price two "
            "programs (declare it without pruning)")
    return shares

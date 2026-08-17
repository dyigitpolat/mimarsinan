"""[wsm V2] Bank-clustered residency: verification and storage dedup of the
planner's streamed composition (the composition itself lives in
``support.schedule.pass_planner`` — one planner, both planes [U1])."""

from __future__ import annotations

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

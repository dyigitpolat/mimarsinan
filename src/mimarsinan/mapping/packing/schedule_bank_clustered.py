"""[wsm V2] Bank-clustered pass composition: banks stay resident, instances stream."""

from __future__ import annotations

from typing import Sequence

from mimarsinan.mapping.packing.softcore import compacted_core_extent
from mimarsinan.mapping.support.schedule.bank_clustered_law import (
    BankInstance,
    compose_bank_clustered_passes,
)
from mimarsinan.mapping.support.schedule.schedule_policy import resident_passes


def try_bank_clustered_passes(
    *,
    cores: list,
    cores_config: Sequence[dict],
    weight_banks: dict,
    max_schedule_passes: int,
) -> "list[list] | None":
    """Compose passes so every physical core keeps one bank while its
    instance queue drains (the weight-stationary regime).

    Instances are sized POST-compaction (:func:`compacted_core_extent`): a
    bank-backed instance carries its elimination as masks until the soft-core
    stage materializes them, so the bank's own matrix is never the extent a
    resident core must budget for.

    Applicability is decided HERE (only this caller can see owned cores and
    intra-segment dependencies); the allocation itself is the shared law in
    ``support.schedule.bank_clustered_law``, so the shape-only layout answer
    search reads composes the SAME passes. Returns per-pass core chunks, or
    ``None`` when the segment is outside this policy's proven class: any owned
    core, any intra-segment dependency, or anything the law declines (an
    instance exceeding every core type, infeasible minimal residency, an
    unstable composition).  Callers fall back to the capacity path.
    """
    del weight_banks  # applicability is structural; sizes live on placements
    if not cores:
        return None
    segment_ids = {core.id for core in cores}
    instances: list[BankInstance] = []
    for core in cores:
        if core.weight_bank_id is None:
            return None
        for source in core.input_sources.flatten():
            if getattr(source, "node_id", None) in segment_ids:
                return None
        axons, neurons = compacted_core_extent(core)
        instances.append(
            BankInstance(
                bank_id=int(core.weight_bank_id), axons=axons, neurons=neurons,
            )
        )

    chunks = compose_bank_clustered_passes(
        instances, cores_config, max_schedule_passes=max_schedule_passes,
    )
    if chunks is None:
        return None
    return [[cores[index] for index in chunk] for chunk in chunks]


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

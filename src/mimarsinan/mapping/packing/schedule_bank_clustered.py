"""[wsm V2] Bank-clustered pass composition: banks stay resident, instances stream."""

from __future__ import annotations

from math import ceil
from typing import Sequence


def try_bank_clustered_passes(
    *,
    cores: list,
    cores_config: Sequence[dict],
    weight_banks: dict,
    max_schedule_passes: int,
) -> "list[list] | None":
    """Compose passes so every physical core keeps one bank while its
    instance queue drains (loads(b) = |C_b|, the weight-stationary regime).

    Returns per-pass core chunks (bank-major, stable order — geometry
    identity across passes follows from identical spec sequences), or
    ``None`` when the segment is outside this policy's proven class:
    any owned core, any intra-segment dependency, an instance exceeding a
    single physical core, more banks than pool cores, or a pass budget
    that cannot be met. Callers fall back to the capacity path.
    """
    del weight_banks  # applicability is structural; sizes live on placements
    if not cores:
        return None
    segment_ids = {core.id for core in cores}
    groups: dict[int, list] = {}
    for core in cores:
        if core.weight_bank_id is None:
            return None
        for source in core.input_sources.flatten():
            if getattr(source, "node_id", None) in segment_ids:
                return None
        groups.setdefault(int(core.weight_bank_id), []).append(core)

    pool_total = sum(int(ct.get("count", 0)) for ct in cores_config)
    if pool_total <= 0 or len(groups) > pool_total:
        return None
    for group in groups.values():
        for core in group:
            axons = len(core.input_sources.flatten())
            neurons = int(core.get_output_count())
            if not any(axons <= int(ct["max_axons"])
                       and neurons <= int(ct["max_neurons"])
                       for ct in cores_config):
                return None

    order = sorted(groups)
    counts = {b: len(groups[b]) for b in order}
    # W_prog-minimizing allocation (§1 of the program): each bank gets the
    # SMALLEST resident core-set that meets the pass budget — loads(b) =
    # |C_b| = ceil(n_b / max_passes); never grow the set just to shave
    # passes (that trades programming bytes for latency, the pool policy's
    # regime). Infeasible residency => fall back to the capacity path.
    budget = max(1, int(max_schedule_passes))
    alloc = {b: max(1, ceil(counts[b] / budget)) for b in order}
    if sum(alloc.values()) > pool_total:
        return None

    passes = max(ceil(counts[b] / alloc[b]) for b in order)
    chunks: list[list] = []
    for index in range(passes):
        chunk: list = []
        for b in order:
            chunk.extend(
                groups[b][index * alloc[b]:(index + 1) * alloc[b]]
            )
        if chunk:
            chunks.append(chunk)
    return chunks


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


def mark_bank_residency(pass_stages: list) -> None:
    """Verify every pass p>0 places, per physical core ordinal, a SUBSET of
    pass 0's (bank, region) placements — regions already programmed, so zero
    new programming (the ragged final pass uses fewer regions). Then mark
    those stages weight-resident. Any divergence is a scheduler bug — fail
    loud, never silently claim residency."""
    if len(pass_stages) <= 1:
        return
    reference = _stage_geometry(pass_stages[0])
    for stage in pass_stages[1:]:
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
        stage.schedule_weights_resident = True

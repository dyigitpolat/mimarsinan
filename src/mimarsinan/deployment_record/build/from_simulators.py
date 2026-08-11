"""Converters: a SANA-FE ``SanafeStepReport`` snapshot → typed record fragments.

Every converter is a pure read of the JSON-safe snapshot shape produced by
``SanafeStepReport.to_snapshot_dict()`` (``chip_simulation/sanafe/stats.py``).
The scalar formulas (``energy_proxy_neuron_steps``, ``latency`` steps, S,
depth) are IMPORTED from ``chip_simulation.cost_extraction`` — the legacy
``CostRecord`` extraction — so the continuity contract
(docs/deployment_record_schema.md §6) holds by construction, never by a
re-derivation that could drift.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

# Continuity SSOT: the private helpers ARE the legacy formulas; importing them
# (rather than copying) makes the §6 value-identity structural.
from mimarsinan.chip_simulation.cost_extraction import (
    _depth_from_sanafe_snapshot,
    _global_s_from_sanafe_snapshot,
    _latency_steps_from_sanafe_snapshot,
    _neuron_steps_from_sanafe_snapshot,
)
from mimarsinan.deployment_record.schema import (
    EnergyRecord,
    EnergyTermRecord,
    FloorplanRecord,
    NocLinkLoadRecord,
    NocTrafficRecord,
    SegmentTimingRecord,
    TileRecord,
)

# The per-event energy planes SANA-FE charges (records/energy.py breakdown).
_ENERGY_BREAKDOWN_TERMS = ("synapse", "dendrite", "soma", "network")
_MEASURED_ENERGY_BASIS = (
    "SANA-FE per-event energy trace, summed over segments and samples"
)


def _sample0_segments(snapshot: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """Sample-0 segment dicts — the legacy extraction's static census sample."""
    per_sample = snapshot.get("per_sample") or []
    if not per_sample:
        raise ValueError(
            "SANA-FE snapshot has no per_sample records; nothing was simulated"
        )
    return list(per_sample[0].get("segments") or [])


def energy_record_from_sanafe(snapshot: Mapping[str, Any]) -> EnergyRecord:
    """The measured energy fragment: aggregate + per-plane breakdown terms.

    ``mj_per_sample`` reproduces ``extract_cost_record`` exactly: the aggregate
    total divided by ``sample_count`` only when more than one sample ran.
    """
    aggregate = snapshot.get("aggregate") or {}
    total_energy_mj = float(aggregate.get("total_energy_mj", 0.0))
    sample_count = int(aggregate.get("sample_count", 0) or 0)
    mj_per_sample = total_energy_mj
    if sample_count > 1:
        mj_per_sample = total_energy_mj / sample_count
    breakdown_j = aggregate.get("energy_breakdown_j") or {}
    breakdown = tuple(
        EnergyTermRecord(
            name=f"sanafe_{term}",
            mj=float(breakdown_j.get(term, 0.0)) * 1000.0,
            kind="measured",
            band_mj=None,
            basis=_MEASURED_ENERGY_BASIS,
        )
        for term in _ENERGY_BREAKDOWN_TERMS
    )
    neuron_steps, _cores = _neuron_steps_from_sanafe_snapshot(snapshot)
    return EnergyRecord(
        total_energy_mj=total_energy_mj,
        mj_per_sample=mj_per_sample,
        sample_count=sample_count,
        breakdown=breakdown,
        energy_proxy_neuron_steps=int(neuron_steps),
        total_spikes=int(aggregate.get("total_spikes", 0)),
    )


def segment_timings_from_sanafe(
    snapshot: Mapping[str, Any]
) -> Tuple[SegmentTimingRecord, ...]:
    """Per-segment measured timing (sample 0 — the legacy latency census).

    ``sim_time_s`` INCLUDES NoC hop latency (charged inside SANA-FE's C++
    NoC model); downstream latency terms must never add hops on top.
    """
    return tuple(
        SegmentTimingRecord(
            stage_index=int(seg["stage_index"]),
            timesteps_executed=int(seg.get("timesteps_executed", 0)),
            sim_time_s=float(seg.get("sim_time_s", 0.0)),
        )
        for seg in _sample0_segments(snapshot)
    )


def s_global_from_sanafe(snapshot: Mapping[str, Any]) -> int:
    """The global temporal resolution S (legacy ``s_global``)."""
    return int(_global_s_from_sanafe_snapshot(snapshot))


def depth_from_sanafe(snapshot: Mapping[str, Any]) -> int:
    """The cascade depth = simulated neural segment count (legacy ``depth``)."""
    return int(_depth_from_sanafe_snapshot(snapshot))


def latency_steps_from_sanafe(snapshot: Mapping[str, Any]) -> int:
    """Σ timesteps_executed over sample-0 segments (legacy ``latency_steps``)."""
    return int(_latency_steps_from_sanafe_snapshot(snapshot))


def noc_traffic_from_sanafe(snapshot: Mapping[str, Any]) -> NocTrafficRecord:
    """The NoC traffic fragment.

    Packet totals sum over EVERY sample and segment (they are event counts);
    ``cross_tile_connectivity_edges`` / ``mapped_cross_tile_axons`` are static
    program properties, so they sum over sample-0 segments only — repeating a
    sample must not multiply a census.
    """
    aggregate = snapshot.get("aggregate") or {}
    inter = intra = input_path = 0
    link_loads: Dict[Tuple[int, int, int, int], int] = {}
    for sample in snapshot.get("per_sample") or []:
        for seg in sample.get("segments") or []:
            inter += int(seg.get("inter_tile_packets", 0))
            intra += int(seg.get("intra_tile_packets", 0))
            input_path += int(seg.get("input_path_packets", 0))
            for load in seg.get("noc_link_load") or []:
                key = (
                    int(load["from_x"]), int(load["from_y"]),
                    int(load["to_x"]), int(load["to_y"]),
                )
                link_loads[key] = link_loads.get(key, 0) + int(
                    load.get("packet_count", 0)
                )
    cross_edges = sum(
        int(seg.get("cross_tile_connectivity_edges", 0))
        for seg in _sample0_segments(snapshot)
    )
    mapped_axons = sum(
        int(seg.get("mapped_cross_tile_axons", 0))
        for seg in _sample0_segments(snapshot)
    )
    return NocTrafficRecord(
        total_packets=int(aggregate.get("total_packets", 0)),
        inter_tile_packets=inter,
        intra_tile_packets=intra,
        input_path_packets=input_path,
        cross_tile_connectivity_edges=cross_edges,
        mapped_cross_tile_axons=mapped_axons,
        link_loads=tuple(
            NocLinkLoadRecord(
                from_x=fx, from_y=fy, to_x=tx, to_y=ty, packet_count=count
            )
            for (fx, fy, tx, ty), count in sorted(link_loads.items())
        ),
    )


def _arch_geometry(snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    """The one arch geometry every segment shares (drift raises)."""
    geometry: Mapping[str, Any] | None = None
    for seg in _sample0_segments(snapshot):
        candidate = seg.get("arch_geometry")
        if candidate is None:
            continue
        if geometry is None:
            geometry = candidate
        elif (
            int(candidate["width"]) != int(geometry["width"])
            or int(candidate["height"]) != int(geometry["height"])
        ):
            raise ValueError(
                "SANA-FE segments disagree on arch geometry: "
                f"{candidate['width']}x{candidate['height']} vs "
                f"{geometry['width']}x{geometry['height']}"
            )
    if geometry is None:
        raise ValueError(
            "SANA-FE snapshot carries no arch_geometry; the floorplan "
            "fragment cannot be built"
        )
    return geometry


def floorplan_from_sanafe(
    snapshot: Mapping[str, Any], *, cores_per_tile: int, derivation: str
) -> FloorplanRecord:
    """The resolved floorplan the run simulated on.

    Mesh dims come from the runner's ``arch_geometry`` — the ``ArchSpec``
    RESOLVED values, where ``mesh_height`` already folds ``floorplan_replicas``
    in (``derive_arch_spec`` row-stacks the physical floorplan k times so every
    logical per-pass core materializes). ``cores_per_tile`` is the declared
    platform's resolved value (``cores_per_tile_resolved``), threaded by the
    caller together with the declared-vs-derived provenance.
    """
    geometry = _arch_geometry(snapshot)
    return FloorplanRecord(
        mesh_width=int(geometry["width"]),
        mesh_height=int(geometry["height"]),
        cores_per_tile=int(cores_per_tile),
        derivation=derivation,
    )


def tiles_from_sanafe(snapshot: Mapping[str, Any]) -> Tuple[TileRecord, ...]:
    """Tile census: the union of per-segment tile occupancies (sample 0)."""
    tiles: Dict[int, Dict[str, Any]] = {}
    for seg in _sample0_segments(snapshot):
        for tile in seg.get("per_tile") or []:
            index = int(tile["tile_index"])
            entry = tiles.setdefault(
                index,
                {"x": int(tile.get("mesh_x", -1)),
                 "y": int(tile.get("mesh_y", -1)),
                 "cores": set()},
            )
            entry["cores"].update(int(c) for c in tile.get("cores") or [])
    return tuple(
        TileRecord(
            tile_index=index,
            x=entry["x"],
            y=entry["y"],
            core_indices=tuple(sorted(entry["cores"])),
        )
        for index, entry in sorted(tiles.items())
    )

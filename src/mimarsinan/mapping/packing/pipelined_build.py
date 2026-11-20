"""[U-F4] Materialize a wavefront segment: each group's passes on ONE fixed load layout.

A pipelined group's layout is the packer's placement of one instance of every copy key of the group. Every pass of
the group is packed from the same ordered soft cores, key by key: the pass's instance of the key, else a stand-in of
the key's template instance (its weights, every axon off). The packer then places every key's region at the same core
ordinal and offsets in every pass; that is verified per core against the group's first pass (else refused by name)
before the stand-ins are stripped. The first pass keeps every layout core (it loads the layout; the cores no instance
of it uses are dark: loaded, not evaluated); every later pass keeps the cores its instances use, each with its head
ordinal, and runs wholly resident. A W02 group of the segment is materialized and verified as W02 materializes it.

[thru:w1] Under declared layer copies the layout holds each key's copies, its cores ordered by key and then copy: one
stand-in per copy (each with ids of its own), and each instance at the copy the plan gives it. A core of a pass that
holds an instance runs wholly: a copy beside it that no instance of the pass uses runs on off axons, its outputs
unread (copies of one key may share a core, and the pass's instances do not always fill them).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from mimarsinan.code_generation.cpp_chip_model_types import SpikeSource
from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.packing.hybrid_segment import _flush_scheduled_segment
from mimarsinan.mapping.packing.hybrid_segment_helpers import _remap_external_sources_to_segment_inputs, _reindex_nodes
from mimarsinan.mapping.packing.infeasibility_proofs import CoreCapacityRefusal, PlannedPassPackingRefusal
from mimarsinan.mapping.packing.neural_segment_packing import neural_segment_to_soft_core_mapping
from mimarsinan.mapping.packing.schedule_bank_clustered import mark_multiplexed_residency, stage_copies
from mimarsinan.mapping.packing.softcore import compact_soft_core_mapping
from mimarsinan.mapping.packing.softcore.sign_split import apply_sign_split
from mimarsinan.mapping.packing.type_binding import refuse_typed_multi_pass
from mimarsinan.mapping.platform.mapping_structure import PositionMultiplexingError
from mimarsinan.mapping.platform.platform_constraints import pool_sign_granularity
from mimarsinan.mapping.support.spike_source_columns import PlacementNeuronMap
from mimarsinan.mapping.support.wavefront_stages import (
    SCHEDULE_CORE_ORDINALS, SCHEDULE_DARK_CORES, SCHEDULE_IDLE_COLUMNS, SCHEDULE_WAVEFRONT, WavefrontHead)

__all__ = ["flush_wavefront_segment", "ir_producers", "refuse_with_retiming"]


def ir_producers(cores: Sequence[Any]) -> List[frozenset]:
    """Per IR core of a segment, the segment positions of the cores it reads (its input sources' nodes)."""
    position = {int(core.id): k for k, core in enumerate(cores)}
    return [frozenset(position[int(s.node_id)] for s in np.asarray(core.input_sources, dtype=object).ravel()
                      if getattr(s, "node_id", None) is not None and int(s.node_id) in position
                      and int(s.node_id) != int(core.id)) for core in cores]


def refuse_with_retiming(pipelined: Any, *, retimed_level_stages: bool, per_hop_neural_segments: bool) -> None:
    """A wavefront pass holds several levels; a build that executes each level as its own unit refuses it by name."""
    if pipelined and (retimed_level_stages or per_hop_neural_segments):
        raise PositionMultiplexingError(
            "pipelined_multiplexing shares passes between levels, and per-hop re-timing (the windowed LIF semantics, "
            "or the per-hop segment split) executes every level as its own unit: refused")


def _special(source: Any) -> bool:
    return bool(source.is_off_ or source.is_input_ or source.is_always_on_)


def _template_standins(nodes, weight_banks, reindex_maps, cores_config, first_id: int) -> List[List[Any]]:
    """Per template node, its soft cores as a pass packs them (compacted, sign split), renumbered from ``first_id``."""
    nodes = _reindex_nodes(list(nodes), reindex_maps) if reindex_maps else list(nodes)
    graph, _inputs = _remap_external_sources_to_segment_inputs(
        nodes=nodes, segment_output_refs=np.array([], dtype=object), weight_banks=weight_banks)
    soft = neural_segment_to_soft_core_mapping(graph, weight_banks)
    compact_soft_core_mapping(soft.cores, soft.output_sources)
    apply_sign_split(soft.cores, pool_sign_granularity(cores_config))
    out: Dict[int, List[Any]] = {}
    for offset, core in enumerate(soft.cores):
        out.setdefault(int(core.id), []).append(core)
        core.id = first_id + offset
    return [out[int(n.id)] for n in nodes]


def _fresh(standin: Any) -> Any:
    """A per-pass copy of a stand-in: its weights shared, every axon a fresh off source."""
    core = copy.copy(standin)
    core.axon_sources = [SpikeSource(-1, 0, is_input=False, is_off=True) for _ in range(len(standin.axon_sources))]
    core._axon_source_spans = None
    return core


def _arranger(keys, node_key: Dict[int, Any], standins: Sequence[Sequence[Any]], label: str):
    """``keys``: the layout's slots in order (a key, or [thru:w1] a key and copy), ``node_key``: per IR node its slot,
    ``standins``: per slot its stand-in soft cores."""
    def arrange(soft_cores: List[Any]) -> List[Any]:
        by_key: Dict[Any, List[Any]] = {}
        for core in soft_cores:
            by_key.setdefault(node_key[int(core.id)], []).append(core)
        ordered: List[Any] = []
        for key, standin in zip(keys, standins):
            ordered.extend(by_key.pop(key, None) or [_fresh(c) for c in standin])
        if by_key:
            raise PositionMultiplexingError(f"pipelined_multiplexing: {label} holds instances of copies its group's "
                                            f"layout does not place ({sorted(map(str, by_key))[:3]})")
        return ordered
    return arrange


def _copy_slots(group, standins: List[List[Any]], first_id: int) -> Tuple[List[Any], List[List[Any]]]:
    """[thru:w1] The layout's slots (key, copy), key by key then copy by copy, and per slot its stand-ins: a copy past
    the first takes its key's stand-ins under ids of its own, numbered on from the first copies'."""
    next_id = first_id + sum(len(cs) for cs in standins)
    slots: List[Any] = []
    per_slot: List[List[Any]] = []
    for key, cores_, count in zip(group.keys, standins, group.copies):
        for c in range(int(count)):
            clones = cores_ if c == 0 else [copy.copy(core) for core in cores_]
            if c:
                for core in clones:
                    core.id, next_id = next_id, next_id + 1
            slots.append((key, c))
            per_slot.append(clones)
    return slots, per_slot


def _keep(stage: Any, standin_ids: frozenset, head: bool, label: str, whole: bool = False) -> None:
    """Strip the stand-ins; a later pass drops the cores they alone held (each keeping its head ordinal), the head
    marks them dark. Every kept core reads only stage inputs (pass transparency), else refused by name. [thru:w1]
    ``whole`` (layer copies): a core that holds an instance beside a stand-in runs wholly, the stand-in on off axons,
    and the pass declares the stand-in's columns idle."""
    hcm = stage.hard_core_mapping
    placements = [[p for p in ps if int(p["ir_node_id"]) not in standin_ids]
                  for ps in hcm.soft_core_placements_per_hard_core]
    placements += [[] for _ in range(len(hcm.cores) - len(placements))]
    if not whole and any(ps and len(ps) != len(full)
                         for ps, full in zip(placements, hcm.soft_core_placements_per_hard_core)):
        raise PositionMultiplexingError(f"pipelined_multiplexing: in {label} a core holds copies its pass uses beside "
                                        "copies it does not; a core of a wavefront pass runs wholly or not at all")
    keep = list(range(len(hcm.cores))) if head else [i for i, ps in enumerate(placements) if ps]
    new = {old: j for j, old in enumerate(keep)}
    idle = tuple((new[i], int(p["neuron_offset"]), int(p["neurons"]))
                 for i, full in enumerate(hcm.soft_core_placements_per_hard_core) if placements[i]
                 for p in full if int(p["ir_node_id"]) in standin_ids)
    if idle:
        setattr(stage, SCHEDULE_IDLE_COLUMNS, idle)
    for core in hcm.cores:
        if any(not _special(s) for s in core.axon_sources):
            raise PositionMultiplexingError(f"pipelined_multiplexing: in {label} a core reads another core of its "
                                            "pass; pass transparency needs every read from an earlier pass")
    for source in np.asarray(hcm.output_sources, dtype=object).flatten():
        if not _special(source):
            source.core_ = new[int(source.core_)]
    hcm.cores = [hcm.cores[i] for i in keep]
    hcm.soft_core_placements_per_hard_core = [placements[i] for i in keep]
    mapping = PlacementNeuronMap()
    for j, ps in enumerate(hcm.soft_core_placements_per_hard_core):
        for p in ps:
            first = (p.get("neuron_range_in_original") or (0,))[0]
            mapping.place(int(p["ir_node_id"]), int(first), int(p["neurons"]), j, int(p["neuron_offset"]))
    hcm.neuron_mapping = mapping
    hcm.unusable_space = sum(int(getattr(c, "unusable_space", 0) or 0) for c in hcm.cores)
    hcm._output_source_spans = None
    if head:
        dark = frozenset(i for i, ps in enumerate(placements) if not ps)
        if dark:
            setattr(stage, SCHEDULE_DARK_CORES, dark)
    else:
        setattr(stage, SCHEDULE_CORE_ORDINALS, tuple(keep))
    ChipLatency(hcm).calculate()  # the pass's own timing: every core reads carried inputs only, so all start at 0


def _materialize(chunk, q, *, label, tag, arrange, consumed_by, cores_config, weight_banks, segment_index,
                 allow_neuron_splitting, allow_coalescing, all_reindex_maps, stages) -> Any:
    nodes = _reindex_nodes(chunk, all_reindex_maps) if all_reindex_maps else chunk
    try:
        built, reindex = _flush_scheduled_segment(
            current_neural=nodes, consumed_by=consumed_by, cores_config=cores_config, weight_banks=weight_banks,
            segment_index=segment_index, segment_label=f"{label}_{tag}{q}", allow_neuron_splitting=allow_neuron_splitting,
            allow_coalescing=allow_coalescing, pass_index=q, arrange=arrange)
    except CoreCapacityRefusal as exc:
        raise PlannedPassPackingRefusal(f"planned pass {q} of segment {label!r} ({len(chunk)} IR cores) does not "
                                        f"pack on the declared cores: {exc}") from exc
    stages.extend(built)
    all_reindex_maps.update(reindex)
    return built[0]


def flush_wavefront_segment(*, plan, specs, spec_to_core, cores, ir_graph, segment_index: int, label: str,
                            **flush) -> None:
    """Materialize every group of ``plan.wavefront`` in pass order; ``flush``: the scheduled flush's keyword arguments
    (consumed_by, cores_config, weight_banks, allow_neuron_splitting, allow_coalescing, all_reindex_maps, stages)."""
    wave = plan.wavefront
    chunks = [[spec_to_core[id(spec)] for spec in chunk] for chunk in plan.pass_lists]
    refuse_typed_multi_pass(label, len(chunks), cores)
    node_key = {int(spec_to_core[id(spec)].id): key for spec, key in zip(specs, wave.keys)}
    if wave.copy_of is not None:  # [thru:w1] each instance at its copy of its key
        node_key = {int(spec_to_core[id(spec)].id): (key, c) for spec, key, c in zip(specs, wave.keys, wave.copy_of)}
    first_id = 1 + max(int(n.id) for n in ir_graph.nodes)
    for g, group in enumerate(wave.groups):
        passes = range(group.first_pass, group.end_pass)
        if not group.pipelined:
            built = [_materialize(chunks[q], q, label=label, tag="cap", arrange=None, segment_index=segment_index,
                                  **flush) for q in passes]
            mark_multiplexed_residency(built, wave.run_heads[group.first_pass:group.end_pass])
            continue
        standins = _template_standins([spec_to_core[id(specs[i])] for i in group.template], flush["weight_banks"],
                                      flush["all_reindex_maps"], flush["cores_config"], first_id)
        slots = group.keys
        if group.copies is not None:
            slots, standins = _copy_slots(group, standins, first_id)
        ids = frozenset(int(c.id) for cs in standins for c in cs)
        arrange = _arranger(slots, node_key, standins, f"segment {label!r}")
        layout = None
        for q in passes:
            stage = _materialize(chunks[q], q, label=label, tag="wave", arrange=arrange, segment_index=segment_index,
                                 **flush)
            copies = stage_copies(stage)
            if layout is None:
                layout = copies
                setattr(stage, SCHEDULE_WAVEFRONT, WavefrontHead(levels=tuple(group.levels), passes=len(passes),
                                                                 layout_cores=len(copies)))
            elif copies != layout:
                raise PositionMultiplexingError(
                    f"pipelined_multiplexing: pass {q} of segment {label!r} places its copies elsewhere than its "
                    "group's first pass loaded them, so it cannot run on that layout")
            stage.schedule_weights_resident = q != group.first_pass
            _keep(stage, ids, q == group.first_pass, f"pass {q} of segment {label!r}", group.copies is not None)

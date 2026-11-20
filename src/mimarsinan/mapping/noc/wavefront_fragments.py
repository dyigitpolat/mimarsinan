"""[U-F4] A wavefront group's passes on the shape-only plane, placed as the builder places them.

The group's layout is ONE pack of its template specs (the first instance of each copy key, in key order); every pass
of the group places each of its instances on its key's cores of that layout. The first pass states every layout core
(it loads them; the cores none of its instances use are its dark cores) and every later pass the layout cores its
instances use, each wholly resident, so the pass programs, the period census and the fit rule read the builder's
program. A W02 group's passes are left to the caller's own packing. [thru:w1] Under layer copies the layout packs each
key's template once per copy (key by key, then copy by copy, ``layout_specs``) and each instance is placed on its own
copy's cores; a core holding an instance runs wholly, as the builder runs it (its other copies idle). [thru:w2] Under
the batch pipeline a pass places every input slot's instances it serves, each on its own copy's cores: the plane's one
pass is the builder's stages of that pass index, one per slot."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from mimarsinan.mapping.layout.layout_packer import pack_layout
from mimarsinan.mapping.noc.pass_program import PassProgram, packed_pass_program
from mimarsinan.mapping.packing.infeasibility_proofs import CoreCapacityRefusal
from mimarsinan.mapping.packing.residency_tiles import CoreTiles
from mimarsinan.mapping.support.pipelined_multiplexing import layout_specs

#: Per planned pass index: its placements ``(program spec index, pass-local core)``, its program and core tiles, and
#: [thru:w3] each placement's input slot under the batch pipeline (None without it).
PlacedPass = Tuple[Tuple[Tuple[int, int], ...], PassProgram, Tuple[CoreTiles, ...], Optional[Tuple[int, ...]]]


def _select(program: PassProgram, keep: Sequence[int]) -> PassProgram:
    """``program`` restricted to its cores ``keep`` (in that order)."""
    def pick(values):
        return tuple(values[i] for i in keep) if values else values

    return replace(program, core_cells=pick(program.core_cells), core_sign_rows=pick(program.core_sign_rows),
                   core_tile_cells=pick(program.core_tile_cells), core_tile_sign_rows=pick(program.core_tile_sign_rows),
                   core_geometry=None if program.core_geometry is None else pick(program.core_geometry),
                   core_used=pick(program.core_used))  # [U-ENERGY] every per-core field follows its core


def wavefront_passes(plan: Any, softcores: Sequence[Any], global_index: Dict[int, int], core_types,
                     allow_neuron_splitting: bool, allow_coalescing: bool, declared_cores,
                     operand_axons: Callable[[Sequence[Any], Any], Tuple[int, ...]]) -> Dict[int, PlacedPass]:
    """Every pass of a wavefront group of ``plan`` (a ``ProgramPassPlan``), by planned pass index."""
    out: Dict[int, PlacedPass] = {}
    offset = 0
    for _sid, segment in plan.segments:
        wave = segment.wavefront
        if wave is not None:
            specs = list(wave.specs)
            for group in wave.groups:
                if group.pipelined:
                    out.update(_group(wave, group, specs, offset, global_index, core_types, allow_neuron_splitting,
                                      allow_coalescing, declared_cores, operand_axons))
        offset += segment.pass_count
    return out


def _group(wave, group, specs, offset, global_index, core_types, allow_neuron_splitting, allow_coalescing,
           declared_cores, operand_axons) -> Dict[int, PlacedPass]:
    template = layout_specs(specs, group.template, group.copies)
    pack = pack_layout(softcores=template, core_types=core_types, allow_neuron_splitting=allow_neuron_splitting,
                       allow_coalescing=allow_coalescing, collect_placements=True)
    if not pack.feasible or pack.placements is None:
        raise CoreCapacityRefusal(f"NoC fragments: a wavefront group's layout does not pack ({pack.error})")
    # [thru:w1] the layout's slots: per key, or per key and copy under layer copies
    slots = list(group.keys) if group.copies is None else [
        (key, c) for key, count in zip(group.keys, group.copies) for c in range(int(count))]
    slot_of = list(wave.keys) if wave.copy_of is None else list(zip(wave.keys, wave.copy_of))
    cores_of: Dict[Any, List[int]] = {}
    for origin, core in pack.placements:
        cores_of.setdefault(slots[int(origin)], []).append(int(core))
    head, tiles = packed_pass_program(pack.used_core_snapshots or (), resident=False, core_types=core_types,
                                      declared_cores=declared_cores, operand_axons=operand_axons(template, pack))
    out: Dict[int, PlacedPass] = {}
    for q in range(group.first_pass, group.end_pass):
        if wave.entry_slots is None:
            placed = [(global_index[id(specs[i])], core) for i in range(len(specs)) if wave.pass_of[i] == q
                      for core in cores_of[slot_of[i]]]
            inputs = None
        else:
            placed, inputs = _entries_placed(wave, q, specs, global_index, cores_of)
        if q == group.first_pass:
            used = {core for _, core in placed}
            dark = frozenset(h for h in range(head.cores) if h not in used)
            out[offset + q] = (tuple(placed), replace(head, wavefront=True, dark_cores=dark or None), tiles, inputs)
            continue
        keep = sorted({core for _, core in placed})
        local = {core: j for j, core in enumerate(keep)}
        program = replace(_select(head, keep), resident=True, wavefront=True, core_ordinals=tuple(keep))
        out[offset + q] = (tuple((spec, local[core]) for spec, core in placed), program,
                           tuple(tiles[h] for h in keep), inputs)
    return out


def _entries_placed(wave, q: int, specs, global_index: Dict[int, int], cores_of: Dict[Any, List[int]]
                    ) -> Tuple[List, Tuple[int, ...]]:
    """[thru:w2] Pass ``q``'s placements under the batch pipeline: every entry (one input slot's instance) on its key's
    cores, or its key's copy's cores under layer copies; [thru:w3] with each placement's input slot."""
    index = {id(spec): i for i, spec in enumerate(specs)}
    copies = wave.entry_copies[q] if wave.entry_copies is not None else (None,) * len(wave.pass_lists[q])
    placed = [(global_index[id(spec)], core, int(b)) for spec, c, b in zip(wave.pass_lists[q], copies,
                                                                          wave.entry_slots[q])
              for core in cores_of[wave.keys[index[id(spec)]] if c is None else (wave.keys[index[id(spec)]], c)]]
    return [(spec, core) for spec, core, _b in placed], tuple(b for _spec, _core, b in placed)

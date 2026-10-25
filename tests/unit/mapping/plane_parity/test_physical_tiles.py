"""[W04 fix round, B1] A fused core of k physical cores is k physical tiles in every priced count, on both planes.

The layout packer keeps a softcore wider than one core whole on one fused core of k = ceil(a / A) members, as the
hard-core packer does (W04 class 1). A fused core is one allocated crossbar but k physical cores: every count the
pricing reads is a count of physical tiles. On the shape-only plane: the packer states each used core's members
(``LayoutCoreSnapshot.tiles``), and the pass programs' segment, reprogrammed and per-pass cores (core initialization, core programming, the shadow room) and the period
census (loaded once is decided on the tiles of each type, one row entry per tile). On the record plane: the
quantities ``segment_cores``, ``reprogrammed_cores`` and ``cores_allocated`` and the period census rows, read off the
sealed core geometry against the declared core types. The two planes agree on every count.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from mimarsinan.deployment_record.build.program_census import (
    candidate_program_passes,
    candidate_programming_census,
    mapping_program_passes,
    record_program_passes,
)
from mimarsinan.deployment_record.build.write_census import WriteCensusError, period_census
from mimarsinan.deployment_record.quantities.from_record import from_record
from mimarsinan.deployment_record.schema import SegmentRecord
from mimarsinan.mapping.crossbar_utilization import CrossbarUtilizationReport
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.latency.ir import IRLatency
from mimarsinan.mapping.layout.layout_packer import pack_layout
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.noc import collect_noc_fragments
from mimarsinan.mapping.packing.hybrid_build_pool import build_hybrid_hard_core_mapping
from mimarsinan.mapping.packing.infeasibility_proofs import CoreCapacityRefusal
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, MappingStrategy
from mimarsinan.mapping.support.residency_rules import spec_runs

from unit.deployment_record.record_fixtures import make_full_record

from . import planes as P


class _Semantics:
    """The executed-window branch ``candidate_program_passes`` duck-types (no compute steps are asked here)."""

    retimed = False
    is_cycle = False
    is_cascade = False
    latency_group_count = None
    chip_latency = None


def _wide(n: int, axons: int = 20, neurons: int = 4):
    return [LayoutSoftCoreSpec(axons, neurons, latency_tag=0, segment_id=0, name=f"w{i}") for i in range(n)]


def _fragments(specs, count: int, *, scheduled: bool, a: int = 8, n: int = 8):
    return collect_noc_fragments(softcores=specs, core_types=[LayoutHardCoreType(a, n, count)], census=None,
                                 allow_scheduling=scheduled, allow_neuron_splitting=False, allow_coalescing=True,
                                 max_schedule_passes=64)


def test_the_packer_states_a_fused_cores_members():
    pack = pack_layout(softcores=_wide(1), core_types=[LayoutHardCoreType(8, 8, 4)], allow_coalescing=True)
    assert pack.feasible and pack.cores_used == len(pack.used_core_snapshots) == 1
    assert pack.used_core_snapshots[0].component_axons == (8, 8, 8)
    assert pack.used_core_snapshots[0].tiles == 3


def test_the_candidate_census_counts_physical_tiles():
    frag = _fragments(_wide(2), 6, scheduled=False)
    assert [p.cores for p in frag.pass_programs] == [2]
    assert [p.tiles for p in frag.pass_programs] == [6]
    prog = candidate_programming_census(frag, weight_bits=8)
    assert (prog.segment_cores, prog.reprogrammed_cores, prog.pass_cores) == (6, 6, (6,))
    assert prog.committed_cells == sum(prog.reprogrammed_cells) == 2 * 20 * 4
    assert prog.reprogrammed_bytes == 2 * 20 * 4


def test_a_fused_core_splits_its_used_rows_over_its_members_in_order():
    frag = _fragments(_wide(1), 3, scheduled=False)
    (program,) = frag.pass_programs
    assert program.tile_cells == (8 * 4, 8 * 4, 4 * 4)
    assert sum(program.tile_cells) == sum(program.core_cells)


def test_the_candidate_period_census_decides_loaded_once_on_physical_tiles():
    """Three fused cores of three members each on six declared cores: three listed cores fit the chip, nine
    physical tiles do not, so the program rewrites every period."""
    frag = _fragments(_wide(3), 6, scheduled=True)
    listed = sum(p.cores for p in frag.pass_programs)
    physical = sum(p.tiles for p in frag.pass_programs)
    assert listed <= 6 < physical == 9
    passes = candidate_program_passes(_wide(3), frag, semantics=_Semantics(), timesteps=None)
    assert sum(len(row.core_cells) for row in passes.rows) == physical
    census = period_census(passes, weight_bits=8, capacity={(8, 8): 6}).census
    assert sum(p.cores for p in census.passes) == physical
    assert sum(p.written_cells_backing_store for p in census.passes) == 3 * 20 * 4
    assert not all(p.resident for p in census.passes)


def _executed(rep, a: int, n: int, count: int, *, scheduled: bool = True, multiplexed: bool = False):
    ir = IRMapping(q_max=1.0, firing_mode="Default", max_axons=a, max_neurons=n, allow_coalescing=True,
                   hardware_bias=True).map(rep)
    rep.clear_ir_caches()
    IRLatency(ir).calculate()
    caps = ChipCapabilities(allow_coalescing=True, allow_neuron_splitting=True, allow_scheduling=scheduled,
                            max_schedule_passes=P.PASSES, position_multiplexing=True if multiplexed else None)
    return build_hybrid_hard_core_mapping(
        ir_graph=ir, cores_config=[{"max_axons": a, "max_neurons": n, "count": int(count), "has_bias": True}],
        strategy=MappingStrategy.resolve(caps), max_schedule_passes=P.PASSES)


def _members(core) -> int:
    return len(getattr(core, "fused_component_axons", None) or [0])


def test_the_build_time_record_plane_states_one_entry_per_member_of_a_declared_type():
    rep = P.representation("conv_odd")
    _, specs = P.walk(rep, 16, 16)
    F = P.footprint(specs, 16, 16)
    hybrid = _executed(rep, 16, 16, F, scheduled=False)
    cores = [c for s in hybrid.stages if s.kind == "neural" for c in s.hard_core_mapping.cores]
    assert any(_members(c) > 1 for c in cores)
    report = CrossbarUtilizationReport.from_hybrid_mapping(hybrid)
    assert (report.cores_allocated, report.physical_cores) == (len(cores), sum(_members(c) for c in cores)) \
        and report.physical_cores == F
    passes = mapping_program_passes(hybrid)
    assert sum(len(row.core_cells) for row in passes.rows) == sum(_members(c) for c in cores)
    assert {t for row in passes.rows for t in row.core_types} == {(16, 16)}
    assert sum(sum(row.core_cells) for row in passes.rows) == sum(
        (c.axons_per_core - c.available_axons) * (c.neurons_per_core - c.available_neurons) for c in cores)
    assert all(p.resident for p in period_census(passes, weight_bits=8, capacity={(16, 16): F}).census.passes)
    # [integration of W04 with W03] The build pins its fit (``mark_fit_residency``): the marks hold all F physical
    # tiles, so on F - 1 cores (more than the allocated crossbars, fewer than the members) they cannot be held and
    # the census refuses by name; counted as allocated crossbars they would have fit.
    assert len(cores) <= F - 1
    with pytest.raises(WriteCensusError, match=f"pin {F} of the {F - 1} cores"):
        period_census(passes, weight_bits=8, capacity={(16, 16): F - 1})


def _fused_record():
    """``make_full_record`` with its first segment's first core a fused core of two declared 256 x 256 cores."""
    record = make_full_record()
    stages = list(record.schedule.stages)
    first = next(i for i, s in enumerate(stages) if isinstance(s, SegmentRecord))
    cores = list(stages[first].cores)
    cores[0] = replace(cores[0], axons=512)
    stages[first] = replace(stages[first], cores=tuple(cores))
    return record, replace(record, schedule=replace(record.schedule, stages=tuple(stages)))


def test_the_sealed_record_counts_a_fused_core_as_its_members():
    plain, fused = _fused_record()
    before, after = from_record(plain), from_record(fused)
    for key in ("segment_cores", "reprogrammed_cores", "cores_allocated"):
        assert after.get(key).value == before.get(key).value + 1, key
    rows = record_program_passes(fused).rows
    assert sum(len(row.core_cells) for row in rows) == sum(len(row.core_cells) for row in
                                                           record_program_passes(plain).rows) + 1
    assert {t for row in rows for t in row.core_types} == {(256, 256)}


def test_a_sealed_geometry_the_declared_types_do_not_determine_withholds_the_counts():
    _, fused = _fused_record()
    platform = dict(fused.identity.platform)
    platform["cores"] = [{"max_axons": 256, "max_neurons": 256, "count": 10},
                         {"max_axons": 128, "max_neurons": 256, "count": 10}]
    ambiguous = replace(fused, identity=replace(fused.identity, platform=platform))
    quantities = from_record(ambiguous)
    for key in ("segment_cores", "reprogrammed_cores", "cores_allocated"):
        assert not quantities.has(key), key
        assert "512 x 256" in quantities.withheld[key]


PARITY = [("conv_odd", 16, 16), ("res", 16, 16), ("wide_fc", 16, 16), ("pw", 16, 8)]


@pytest.mark.parametrize("name,a,n", PARITY, ids=[f"{p[0]}-{p[1]}x{p[2]}" for p in PARITY])
@pytest.mark.parametrize("multiplexed", [False, True], ids=["replicated", "multiplexed"])
def test_both_planes_count_the_same_physical_tiles(name, a, n, multiplexed):
    rep = P.representation(name)
    layout, specs = P.walk(rep, a, n)
    F = P.footprint(specs, a, n)
    caps = ChipCapabilities(allow_coalescing=True, allow_neuron_splitting=True, allow_scheduling=True,
                            max_schedule_passes=P.PASSES, position_multiplexing=True if multiplexed else None)
    fused_seen = False
    for count in sorted({F, math.ceil(F / 2), math.ceil(F / 8)}):
        try:
            frag = collect_noc_fragments(softcores=specs, core_types=[LayoutHardCoreType(a, n, count)], census=None,
                                         runs=spec_runs(layout), **caps.layout_kwargs())
        except CoreCapacityRefusal:  # both planes refuse a count below one softcore's window (test_planes)
            with pytest.raises(CoreCapacityRefusal):
                _executed(rep, a, n, count, multiplexed=multiplexed)
            continue
        shape = candidate_programming_census(frag, weight_bits=8)
        hybrid = _executed(rep, a, n, count, multiplexed=multiplexed)
        stages = [s for s in hybrid.stages if s.kind == "neural"]
        tiles = [sum(_members(c) for c in s.hard_core_mapping.cores) for s in stages]
        fused_seen = fused_seen or any(_members(c) > 1 for s in stages for c in s.hard_core_mapping.cores)
        resident = [bool(s.schedule_weights_resident) for s in stages]
        assert [p.resident for p in frag.pass_programs] == resident, count
        assert list(shape.pass_cores) == tiles, count
        assert shape.segment_cores == sum(tiles), count
        assert shape.reprogrammed_cores == sum(t for t, r in zip(tiles, resident) if not r), count
        assert [len(row.core_cells) for row in mapping_program_passes(hybrid).rows] == tiles, count
    assert fused_seen

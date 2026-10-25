"""G3's carried CL3 item (``PassProgram.core_geometry``, G5): every pass program of the shape-only plane states the
geometry of each core it packs, so the candidate plane's period census places its cores on a chip of several core
types (before, an untyped plane on such a chip placed no rewritten core)."""

from __future__ import annotations

from mimarsinan.mapping.support.core_slots import stated_tiles
from mimarsinan.deployment_record.build.program_census import candidate_program_passes
from mimarsinan.deployment_record.build.write_census import core_capacity_of
from mimarsinan.deployment_record.quantities.write_law import plane_write_census
from mimarsinan.mapping.layout.layout_types import core_types_from_declaration
from mimarsinan.mapping.noc import collect_noc_fragments
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities
from mimarsinan.search.problems.joint.candidate_fragments import StageSemantics

from .executed import templates as TV


def _fragments(scale: float, plan=None):
    vehicle = TV.template_vehicle("t0_44")
    platform = {**TV.residency_platform(vehicle, scale, plan), "weight_bits": 8}
    fragments = collect_noc_fragments(softcores=vehicle.specs, core_types=core_types_from_declaration(platform["cores"]),
                                      census=vehicle.census, program=vehicle.program,
                                      **ChipCapabilities.from_platform_constraints(platform).layout_kwargs())
    return vehicle, platform, fragments


def test_every_pass_program_states_each_cores_geometry():
    _vehicle, platform, fragments = _fragments(4.0)
    capacity = core_capacity_of(platform)
    assert len(capacity) == 2                                       # t0_44's chip: 784 x 512 and 512 x 256 cores
    for program in fragments.pass_programs:
        assert len(program.core_geometry) == program.cores
        assert all(stated_tiles(tuple(g), capacity) is not None for g in program.core_geometry)


def _census(plan, scale: float = 4.0):
    vehicle, platform, fragments = _fragments(scale, plan)
    passes = candidate_program_passes(vehicle.specs, fragments, semantics=StageSemantics(), timesteps=1)
    census, reason = plane_write_census(passes, platform, weight_bits=8, timesteps=1, label="candidate")
    assert census is not None, reason
    assert all(row.core_types is not None for row in passes.rows)
    assert "no core type" not in census.census.label
    return census


def test_the_candidate_census_places_its_cores_on_a_chip_of_two_core_types():
    """The stateless plan rewrites every core each period, so its census places cores of both declared types."""
    assert {slot.core_type for slot in _census({"step": "stateless"}).slot_writes} == {(784, 512), (512, 256)}


def test_without_the_key_the_passes_fit_side_by_side_and_nothing_is_rewritten():
    """[W03] At x4 the passes that load their own cores (each segment's first, U1's bank-clustered composition)
    fit the declared pool side by side, so the planner pins the program and the census places no rewrite."""
    _vehicle, _platform, fragments = _fragments(4.0)
    assert all(p.schedule_resident_cores == frozenset(range(p.cores)) for p in fragments.pass_programs)
    census = _census(None)
    assert not census.slot_writes and sum(p.payload_bytes for p in census.census.passes) == 0


def test_the_floor_s_pins_fit_by_dominance_and_the_census_is_stated():
    """t0_44 at x3.4 (40 + 40 cores, below the 42 + 42 on which its unscheduled pack holds) under the floor: the
    512 x 256 pins and a pass's 512 x 256 tiles exceed that type's 40 cores; the builder fits the excess on idle
    784 x 512 cores, so the census is stated with no rewrite placed. [fix_c] At x4 (48 + 48) the fit rule judges the
    floor too: its unscheduled pack holds, so the floor is that pack, pinned, and nothing is rewritten."""
    census = _census({"step": "floor"}, 3.4)
    assert census.slot_writes is None and "at least its size" in census.census.label
    _vehicle, _platform, fragments = _fragments(4.0, {"step": "floor"})
    assert all(p.schedule_resident_cores == frozenset(range(p.cores)) for p in fragments.pass_programs)
    fit = _census({"step": "floor"})
    assert not fit.slot_writes and sum(p.payload_bytes for p in fit.census.passes) == 0

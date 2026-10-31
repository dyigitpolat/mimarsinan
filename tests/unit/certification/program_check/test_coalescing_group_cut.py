"""[U-CG] H9 reads the planner's coalescing rule: a coalescing group whose members each hold their whole fan-in (every
placement's role ``master``) may run in several passes, each member in one; a member cut across passes, or a cut
group with a member of another role (a partial sum), is still ``coalescing_group_cut``.

The vehicle: G5's chain with a wide middle layer (widths 32, 16, 6 on 25 x 8 cores, coalescing declared): the
16-neuron layer reads 32 inputs, so it is two output tiles of two fused cores each, one group of four cores. At three
and two cores the group alone exceeds one pass and the builder plans it between its members; the static check then
establishes Theorem 1 on the split program under the streamed and the windowed semantics."""

from __future__ import annotations

import copy

import pytest

from unit.certification.program_check import programs as P
from unit.certification.program_check.test_clean_programs import _established
from unit.certification.program_check.test_mutants import _failed, _units
from unit.mapping.residency.executed import vehicles as V

WIDTHS = (32, 16, 6)


def _split(count: int, *, retimed: bool = False):
    ir = P.integer_ir(V.chain_repr(0, widths=WIDTHS), coalescing=True)
    program = V.build(ir, count, None, retimed_level_stages=retimed, allow_coalescing=True,
                      allow_neuron_splitting=True)
    return program, ir


def _group_units(program):
    """Per coalescing group, per member node, the units holding its placements; and the group's roles."""
    units, roles = {}, {}
    for u, unit in enumerate(_units(program)):
        for records in unit.hard_core_mapping.soft_core_placements_per_hard_core:
            for r in records:
                gid = r.get("coalescing_group_id")
                if gid is not None:
                    units.setdefault(gid, {}).setdefault(int(r["ir_node_id"]), set()).add(u)
                    roles.setdefault(gid, set()).add(r.get("coalescing_role"))
    return units, roles


@pytest.mark.parametrize("count", [2, 3])
def test_the_group_runs_in_several_passes_each_member_in_one(count):
    program, _ = _split(count)
    units, roles = _group_units(program)
    assert len(units) == 1
    (members,) = units.values()
    assert len(members) == 2 and all(len(u) == 1 for u in members.values())
    assert len(set().union(*members.values())) == 2
    assert roles == {next(iter(units)): {"master"}}


@pytest.mark.parametrize("count", [2, 3])
def test_a_streamed_split_program_establishes_theorem1(count):
    program, ir = _split(count)
    report = P.run(program, ir, P.decl("streamed_lif", count=count))
    _established(report, "tiles", "H6", "H9", "R6")


@pytest.mark.parametrize("count", [2, 3])
def test_a_windowed_split_program_establishes_theorem1(count):
    program, ir = _split(count, retimed=True)
    report = P.run(program, ir, P.decl("windowed_lif", count=count), identity=P.identity_of(ir, retimed=True))
    _established(report, "tiles", "R5", "H0", "H9")


def _records(program, gid):
    for unit in _units(program):
        for records in unit.hard_core_mapping.soft_core_placements_per_hard_core:
            for r in records:
                if r.get("coalescing_group_id") == gid:
                    yield unit, r


@pytest.mark.parametrize("roles", [("partial", "accum"), ("master", "partial"), ("master", None)])
def test_a_cut_group_with_a_member_not_whole_is_refused_by_name(roles):
    program, ir = _split(3)
    bad = copy.deepcopy(program)
    units, _ = _group_units(bad)
    (gid,) = units
    nodes = sorted(units[gid])
    for _, record in _records(bad, gid):
        record["coalescing_role"] = roles[nodes.index(int(record["ir_node_id"]))]
    _failed(P.run(bad, ir, P.decl("streamed_lif", count=3)), "H9", "coalescing_group_cut")


def test_a_member_cut_across_passes_is_refused_by_name():
    """A whole-fan-in member whose placements run in two passes: a placement of another core in another pass is
    relabelled as a part of it."""
    program, ir = _split(3)
    bad = copy.deepcopy(program)
    units, _ = _group_units(bad)
    (gid,) = units
    member = min(units[gid])
    home = next(iter(units[gid][member]))
    other = next(r for u, unit in enumerate(_units(bad)) if u != home
                 for records in unit.hard_core_mapping.soft_core_placements_per_hard_core for r in records
                 if r.get("coalescing_group_id") is None)
    other.update(ir_node_id=member, coalescing_group_id=gid, coalescing_role="master")
    report = P.run(bad, ir, P.decl("streamed_lif", count=3))
    _failed(report, "H9", "coalescing_group_cut")

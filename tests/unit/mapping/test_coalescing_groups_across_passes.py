"""[U-CG] Coalescing groups across passes: a coalescing group whose members each hold their whole fan-in (fused cores,
role ``master``) shares no partial sum, so when it alone exceeds one pass the capacity split plans it as its members,
each one atomic bundle like any ungrouped spec, never cut inside one. A group that packs alone stays one bundle, as
before, and a group with any other member (a partial-sum member, or a member whose role is unknown) keeps the one-pass
rule: one infeasible bundle, refused by name. The rule is reached only where a group alone does not pack, so no point
the split admitted before plans differently; the reference splitter (the oracle) takes the same decisions.

The witnesses: three output tiles of a 300-input layer on 128 x 128 crossbars (three fused crossbars each, one group
of nine) on pools of four to eight crossbars, and seeded segment families with grouped specs."""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from mimarsinan.mapping.layout.layout_packer import pack_layout
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.support.schedule.schedule_partitioner import estimate_passes_for_layout_validated
from mimarsinan.mapping.support.schedule.schedule_split import split_softcores_by_capacity, split_with_verdicts

from unit.mapping.packing import reference_engine as ref

A = N = 128
MASTER = "master"


def _tile(neurons: int, name: str, *, gid=0, role=MASTER, lat=0, axons: int = 300) -> LayoutSoftCoreSpec:
    return LayoutSoftCoreSpec(axons, neurons, latency_tag=lat, name=name, coalescing_group_id=gid,
                              coalescing_role=role)


def _group(role=MASTER):
    return [_tile(128, "t0", role=role), _tile(128, "t1", role=role), _tile(44, "t2", role=role)]


def _ids(subs):
    return [[id(s) for s in sub] for sub in subs]


def _pool(count: int):
    return [LayoutHardCoreType(A, N, count)]


class TestAWholeFanInGroupAboveOnePass:
    def test_it_splits_between_members_each_whole(self):
        specs = _group()
        out = split_with_verdicts(specs, _pool(4), allow_coalescing=True, coalescing_group_ids=[0, 0, 0])
        assert all(ok for _, ok in out), [(len(sub), ok) for sub, ok in out]
        assert _ids(sub for sub, _ in out) == [[id(specs[0])], [id(specs[1])], [id(specs[2])]]
        for sub, _ in out:
            assert pack_layout(softcores=sub, core_types=_pool(4), allow_coalescing=True).feasible

    @pytest.mark.parametrize("count", [3, 4, 5, 6, 7, 8])
    def test_its_members_accumulate_as_ungrouped_specs_do(self, count):
        """The members are planned exactly as the same specs without a group: the plan the shape-only plane priced
        before the planner saw coalescing groups (W04 class 4), with each member whole."""
        lead = LayoutSoftCoreSpec(100, 128, latency_tag=0, name="lead")
        tail = LayoutSoftCoreSpec(200, 10, latency_tag=1, name="tail")
        specs = [lead, *_group(), tail]
        gids = [None, 0, 0, 0, None]
        got = split_with_verdicts(specs, _pool(count), allow_coalescing=True, coalescing_group_ids=gids)
        free = split_with_verdicts(specs, _pool(count), allow_coalescing=True, coalescing_group_ids=[None] * 5)
        assert [([id(s) for s in sub], ok) for sub, ok in got] == [([id(s) for s in sub], ok) for sub, ok in free]
        assert all(ok for _, ok in got)

    def test_the_validated_estimate_plans_it(self):
        specs = _group()
        n, passes, ok = estimate_passes_for_layout_validated(specs, 4, core_types=_pool(4), allow_coalescing=True,
                                                             coalescing_group_ids=[0, 0, 0])
        assert ok and n == 3 and _ids(passes) == [[id(s)] for s in specs]

    def test_a_member_that_alone_exceeds_the_pool_is_still_refused(self):
        """Never inside a member: a member needing three fused crossbars on a pool of two is one infeasible pass."""
        specs = _group()
        out = split_with_verdicts(specs, _pool(2), allow_coalescing=True, coalescing_group_ids=[0, 0, 0])
        assert not all(ok for _, ok in out)
        assert sorted(len(sub) for sub, _ in out) == [1, 1, 1]


class TestAGroupThatPacksAloneKeepsOnePass:
    @pytest.mark.parametrize("count", [9, 12])
    def test_one_bundle(self, count):
        specs = _group()
        out = split_with_verdicts(specs, _pool(count), allow_coalescing=True, coalescing_group_ids=[0, 0, 0])
        assert _ids(sub for sub, _ in out) == [[id(s) for s in specs]] and all(ok for _, ok in out)


class TestTheScopeGuard:
    def test_a_group_without_roles_keeps_one_infeasible_bundle(self):
        """A partial-plus-accumulator group as ``test_coalescing_schedule_split`` builds it (no roles), sized so no two
        of its specs share a core and three exceed the pool of two: the group stays one bundle and the pass is
        infeasible (refused by name)."""
        specs = [LayoutSoftCoreSpec(30, 20, residency_class_id=1, latency_tag=0, name="p0"),
                 LayoutSoftCoreSpec(30, 20, residency_class_id=1, latency_tag=0, name="p1"),
                 LayoutSoftCoreSpec(20, 20, residency_class_id=1, latency_tag=0, name="acc")]
        out = split_with_verdicts(specs, [LayoutHardCoreType(32, 32, 2)], allow_coalescing=True,
                                  coalescing_group_ids=[0, 0, 0])
        assert _ids(sub for sub, _ in out) == [[id(s) for s in specs]]
        assert [ok for _, ok in out] == [False]

    @pytest.mark.parametrize("roles", [("partial", "partial", "accum"), (MASTER, MASTER, "accum"),
                                       (MASTER, None, MASTER)])
    def test_a_group_with_a_member_not_whole_keeps_one_infeasible_bundle(self, roles):
        specs = [_tile(n, f"t{i}", role=r) for i, (n, r) in enumerate(zip((128, 128, 44), roles))]
        out = split_with_verdicts(specs, _pool(4), allow_coalescing=True, coalescing_group_ids=[0, 0, 0])
        assert _ids(sub for sub, _ in out) == [[id(s) for s in specs]] and [ok for _, ok in out] == [False]

    def test_undeclared_coalescing_reads_no_group(self):
        specs = _group()
        got = split_softcores_by_capacity(specs, _pool(4), allow_coalescing=False, coalescing_group_ids=[0, 0, 0])
        assert _ids(got) == _ids(split_softcores_by_capacity(specs, _pool(4), allow_coalescing=False))


def _family(seed: int):
    """A seeded segment with coalescing groups: each group's members are whole-fan-in tiles of one width (role
    ``master``), or, in a share of the groups, carry another role; pools of one core shape, some of them small."""
    rng = random.Random(seed)
    specs, gids, gid = [], [], 0
    for i in range(rng.randint(1, 14)):
        lat = rng.choice([None, 0, 1, 2])
        if rng.random() < 0.45:
            axons = rng.choice([129, 200, 300, 400, 600])
            members = rng.randint(2, 5)
            other = rng.random() < 0.25
            for m in range(members):
                role = MASTER if not other or m else rng.choice(["partial", "accum", None])
                specs.append(LayoutSoftCoreSpec(axons, rng.choice([128, 64, 44]), latency_tag=lat, name=f"g{i}_{m}",
                                                coalescing_group_id=gid, coalescing_role=role,
                                                residency_class_id=rng.choice([0, 1])))
                gids.append(gid)
            gid += 1
        else:
            specs.append(LayoutSoftCoreSpec(rng.choice([16, 64, 128, 100]), rng.choice([16, 64, 128]), latency_tag=lat,
                                            name=f"c{i}", residency_class_id=rng.choice([0, 1, 2])))
            gids.append(None)
    types = [LayoutHardCoreType(A, N, rng.choice([2, 3, 4, 5, 6, 8, 12, 24, 48, 96]))]
    return specs, gids, types, dict(allow_coalescing=True, allow_splitting=rng.random() < 0.5)


def _roleless(specs):
    return [replace(s, coalescing_role=None) for s in specs]


class TestNoAdmittedPlanMoves:
    def test_every_plan_the_one_pass_rule_admits_is_unchanged(self):
        """Where the plan with every group held in one pass (the roles cleared) is feasible, the plan with the roles is
        the same plan; the roles move only plans that rule refuses."""
        admitted = moved = 0
        for seed in range(400):
            specs, gids, types, kw = _family(seed)
            got = split_with_verdicts(specs, types, coalescing_group_ids=gids, **kw)
            plain = _roleless(specs)
            before = split_with_verdicts(plain, types, coalescing_group_ids=gids, **kw)
            index = {id(p): i for i, p in enumerate(plain)}
            mine = {id(s): i for i, s in enumerate(specs)}
            as_before = [([index[id(s)] for s in sub], ok) for sub, ok in before]
            as_now = [([mine[id(s)] for s in sub], ok) for sub, ok in got]
            if all(ok for _, ok in before):
                admitted += 1
                assert as_now == as_before, seed
            elif as_now != as_before:
                moved += 1
        assert admitted > 100 and moved > 20, (admitted, moved)


class TestTheReferenceTakesTheSameDecisions:
    def test_sub_segments_equal(self):
        for seed in range(300):
            specs, gids, types, kw = _family(seed)
            got = split_softcores_by_capacity(specs, types, coalescing_group_ids=gids, **kw)
            want = ref.reference_split_softcores_by_capacity(specs, types, coalescing_group_ids=gids, **kw)
            assert _ids(got) == _ids(want), seed

    def test_the_validated_estimate_equals_the_reference(self):
        for seed in range(150):
            specs, gids, types, kw = _family(seed)
            budget = sum(t.count for t in types)
            got = estimate_passes_for_layout_validated(specs, budget, core_types=types, coalescing_group_ids=gids, **kw)
            want = ref.reference_estimate_passes_for_layout_validated(specs, budget, core_types=types,
                                                                      coalescing_group_ids=gids, **kw)
            assert (got[0], _ids(got[1]), got[2]) == (want[0], _ids(want[1]), want[2]), seed

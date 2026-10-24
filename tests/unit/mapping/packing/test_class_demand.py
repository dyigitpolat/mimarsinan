"""U1 planner scaling, review round 1: ``pack_layout``'s verdict on a pool of one core shape, by class demands.

The theorem (``packing/class_demand.py``): on a pool of one core shape each residency class's greedy run reads only
its own cores, and on an unbounded pool it consumes a fixed number of pool cores, its demand; a run on ``count``
cores follows the unbounded run until a step the pool cannot serve, where it raises. So a batch packs iff no class's
run raises and the demands sum to at most ``count``. The oracle is ``reference_engine.reference_pack_layout``, the
packer before U1 planner scaling. The split's uncertified regime (a pool below its segment, several softcores per
core) is decided this way, a bundle at a time.
"""

from __future__ import annotations

import random

import pytest

import mimarsinan.mapping.layout.layout_packer as lp
import mimarsinan.mapping.packing.class_demand as cd
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.packing.class_demand import UNPLACEABLE, PoolDemand
from mimarsinan.mapping.support.schedule.schedule_partitioner import estimate_passes_for_layout_validated
from mimarsinan.mapping.support.schedule.schedule_split import split_softcores_by_capacity

from unit.mapping.packing import reference_engine as ref

#: Shapes many of which share a core, shapes a neuron split or a fusion places, and coalescing fragments of one and
#: of two engine priorities (600 axons: three of 200; 257: one of 129 and one of 128).
PALETTE = [(16, 16), (16, 64), (32, 32), (8, 100), (100, 8), (64, 16), (144, 16), (0, 5), (5, 0), (16, 96),
           (96, 96), (40, 300), (300, 40), (257, 16), (600, 16), (16, 600), (128, 200)]


def _specs(rng, n):
    palette = rng.sample(PALETTE, rng.randint(1, 4))
    classes = rng.choice([1, 2, 3, 6])
    out = [LayoutSoftCoreSpec(*(rng.choice(palette) if rng.random() < 0.9 else
                                (rng.randint(0, 600), rng.randint(0, 300))),
                              residency_class_id=rng.randrange(classes),
                              latency_tag=rng.choice([None, 0, 1, 2, 7]),
                              name=None if rng.random() < 0.5 else f"p{rng.randrange(5)}",
                              # [W04] banks (bank affinity) and coalescing groups (never split): what the packer reads
                              bank_id=rng.choice([None, None, 0, 1, 2]),
                              coalescing_group_id=rng.choice([None, None, None, 0]))
           for _ in range(n)]
    if rng.random() < 0.5:
        out.sort(key=lambda s: s.latency_tag or 0)
    return out


def _types(rng):
    a, n = rng.choice([(256, 256), (128, 128), (256, 64), (64, 256), (100, 64)])
    types = [LayoutHardCoreType(a, n, rng.choice([1, 2, 3, 5, 8, 13, 20, 40]))]
    roll = rng.random()
    if roll < 0.25:
        types.append(LayoutHardCoreType(rng.choice([2 * a, a + 1, 600]), rng.choice([n, 2 * n]), 0))
    elif roll < 0.4:
        types.append(LayoutHardCoreType(a, n, rng.choice([0, 1, 4])))
    return types


def _unbounded_consumption(items, types, split, coalesce):
    """The pool cores the reference pack consumes on a pool larger than any run of ``items`` can use."""
    a, n = next((int(t.max_axons), int(t.max_neurons)) for t in types if int(t.count) > 0)
    enough = sum((-(-int(s.input_count) // a) + 1) * (-(-int(s.output_count) // n) + 1) for s in items)
    pool = [LayoutHardCoreType(a, n, enough)] + [LayoutHardCoreType(t.max_axons, t.max_neurons, 0) for t in types]
    result = ref.reference_pack_layout(softcores=items, core_types=pool, allow_neuron_splitting=split,
                                       allow_coalescing=coalesce)
    return sum(s.axons_per_core // a for s in result.used_core_snapshots) if result.feasible else UNPLACEABLE


MODES = [(s, c) for s in (False, True) for c in (False, True)]


class TestTheVerdictIsThePackersVerdict:
    @pytest.mark.parametrize("split,coalesce", MODES)
    def test_feasibility_equals_the_reference_pack(self, split, coalesce):
        for seed in range(70):
            rng = random.Random(seed)
            specs, types = _specs(rng, rng.randint(0, 60)), _types(rng)
            demand = PoolDemand.for_pool(specs, types, allow_coalescing=coalesce, allow_splitting=split)
            assert demand is not None
            for batch in [specs] + [rng.sample(specs, rng.randint(0, len(specs))) for _ in range(2)]:
                want = ref.reference_pack_layout(softcores=batch, core_types=types, allow_neuron_splitting=split,
                                                 allow_coalescing=coalesce).feasible
                assert demand.feasible(batch) == want, seed

    @pytest.mark.parametrize("split,coalesce", MODES)
    def test_a_class_demand_is_what_its_unbounded_run_consumes(self, split, coalesce):
        """Single-type classes of every size up to 70 (the prefix tables, a neuron split included) and mixed ones."""
        types = [LayoutHardCoreType(256, 256, 1)]
        demand = PoolDemand.for_pool([], types, allow_coalescing=coalesce, allow_splitting=split)
        assert demand is not None
        for a, n in PALETTE:
            for k in (1, 2, 3, 5, 17, 33, 70):
                items = [LayoutSoftCoreSpec(a, n, latency_tag=0) for _ in range(k)]
                runs = (((a, n, 0, None, False), k),)
                assert demand.classes.demand(runs) == _unbounded_consumption(items, types, split, coalesce), (a, n, k)
        rng = random.Random(7)
        for _ in range(40):
            items = [LayoutSoftCoreSpec(*rng.choice(PALETTE), latency_tag=rng.choice([None, 0, 3]))
                     for _ in range(rng.randint(1, 30))]
            runs = cd._ClassRuns().extended(items)[0]
            assert demand.classes.demand(runs) == _unbounded_consumption(items, types, split, coalesce)

    def test_inputs_outside_the_theorem_are_not_decided(self):
        types = [LayoutHardCoreType(256, 256, 4)]
        spec = LayoutSoftCoreSpec(16, 16)
        assert PoolDemand.for_pool([spec, spec], types, allow_coalescing=False, allow_splitting=False) is None
        assert PoolDemand.for_pool([spec], types + [LayoutHardCoreType(128, 128, 1)], allow_coalescing=False,
                                   allow_splitting=False) is None
        assert PoolDemand.for_pool([LayoutSoftCoreSpec(16.0, 16)], types, allow_coalescing=False,
                                   allow_splitting=False) is None
        assert PoolDemand.for_pool([spec], [LayoutHardCoreType(256, 256, 0)], allow_coalescing=False,
                                   allow_splitting=False) is None


class TestTheRunningTallyIsExact:
    @pytest.mark.parametrize("split,coalesce", MODES)
    def test_each_extension_answers_as_the_pack_of_the_running_batch_does(self, split, coalesce):
        for seed in range(25):
            rng = random.Random(100 + seed)
            specs, types = _specs(rng, rng.randint(1, 80)), _types(rng)
            demand = PoolDemand.for_pool(specs, types, allow_coalescing=coalesce, allow_splitting=split)
            assert demand is not None
            tally, running = demand.running(), []
            for sc in specs:
                if rng.random() < 0.1:
                    running = running[: rng.randint(0, len(running))]
                    tally.reset(running)
                want = ref.reference_pack_layout(softcores=running + [sc], core_types=types,
                                                 allow_neuron_splitting=split, allow_coalescing=coalesce).feasible
                assert tally.extends([sc]) == want, seed
                if want:
                    running.append(sc)


def _segment(seed):
    """A segment far above its pool with several softcores per core: the capacity certificate decides little."""
    rng = random.Random(seed)
    specs = _specs(rng, rng.randint(20, 150))
    for i, sc in enumerate(specs):
        if rng.random() < 0.5:
            specs[i] = LayoutSoftCoreSpec(sc.input_count, sc.output_count, residency_class_id=sc.residency_class_id,
                                          latency_tag=i * 4 // len(specs), name=sc.name)
    types = [LayoutHardCoreType(256, 256, rng.choice([1, 2, 3, 6, 10]))]
    groups = None
    if rng.random() < 0.3:
        groups, gid = [], 0
        for _ in specs:
            groups.append(gid if rng.random() < 0.4 else None)
            gid += rng.random() < 0.5
    return specs, types, groups, rng


class TestTheSplitBelowTheSegment:
    def test_sub_segments_equal_the_reference(self):
        for seed in range(90):
            specs, types, groups, rng = _segment(seed)
            kw = dict(allow_coalescing=rng.random() < 0.6, allow_splitting=rng.random() < 0.5,
                      coalescing_group_ids=groups)
            got = split_softcores_by_capacity(specs, types, **kw)
            want = ref.reference_split_softcores_by_capacity(specs, types, **kw)
            assert [[id(s) for s in sub] for sub in got] == [[id(s) for s in sub] for sub in want], seed

    def test_the_validated_estimate_equals_the_reference(self):
        for seed in range(60):
            specs, types, groups, rng = _segment(500 + seed)
            kw = dict(allow_coalescing=rng.random() < 0.6, allow_splitting=rng.random() < 0.5, core_types=types,
                      coalescing_group_ids=groups)
            got = estimate_passes_for_layout_validated(specs, sum(t.count for t in types), **kw)
            want = ref.reference_estimate_passes_for_layout_validated(specs, sum(t.count for t in types), **kw)
            assert (got[0], [[id(s) for s in p] for p in got[1]], got[2]) == \
                (want[0], [[id(s) for s in p] for p in want[1]], want[2]), seed


class TestTheCostBelowTheSegment:
    @pytest.mark.parametrize("split", [False, True])
    def test_a_segment_far_above_its_pool_packs_a_linear_number_of_softcores(self, monkeypatch, split):
        """3,000 softcores in three classes and latency groups, 16 per core, on 40 cores: the reference packs every
        accumulation step from scratch, about 3,000 x 320 softcores; the split packs only the class demand tables.
        A pass closes on the pool's 40 cores: 640 of class 0; 360 of class 0 (23 cores) and 272 of class 1; ..."""
        packed = {"n": 0}
        original = lp.pack_layout

        def counting(**kw):
            packed["n"] += len(kw["softcores"])
            return original(**kw)

        monkeypatch.setattr(lp, "pack_layout", counting)   # the class demands measure with this packer, afresh
        n = 3000
        specs = [LayoutSoftCoreSpec(16, 16, residency_class_id=i * 3 // n, latency_tag=i * 3 // n) for i in range(n)]
        out = split_softcores_by_capacity(specs, [LayoutHardCoreType(256, 256, 40)], allow_splitting=split)
        assert [len(sub) for sub in out] == [640, 632, 640, 632, 456]
        assert packed["n"] <= 8 * n

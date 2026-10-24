"""U1 planner scaling: the capacity split takes exactly the reference split's decisions, packs nothing a capacity
certificate already decides, and the validated estimator reuses the split's verdicts instead of packing again.

The certificate: on one core shape (A, N) with ``count`` cores, a softcore (after the packer's axon-coalescing
expansion) opens at most 1 core when it fits a fresh one, ceil(a / A) when it is fused, ceil(n / N) when it is
neuron-split, and is unplaceable otherwise; when the bounds of a set sum to at most ``count``, the greedy packs
that set and every subset of it. The oracle is ``unit.mapping.packing.reference_engine``.
"""

from __future__ import annotations

import random

import pytest

import mimarsinan.mapping.layout.layout_packer as lp
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.support.schedule.schedule_partitioner import estimate_passes_for_layout_validated
from mimarsinan.mapping.support.schedule.schedule_split import capacity_bound, split_softcores_by_capacity

from unit.mapping.packing import reference_engine as ref


def _segment(seed, *, homogeneous=None):
    rng = random.Random(seed)
    palette = [(144, 16), (96, 16), (64, 16), (16, 16), (16, 64), (300, 40), (40, 300), (600, 16), (0, 3)]
    specs = [LayoutSoftCoreSpec(*(rng.choice(palette) if rng.random() < 0.8 else
                                  (rng.randint(0, 700), rng.randint(0, 400))),
                                residency_class_id=rng.choice([0, 1, 2]), latency_tag=rng.choice([None, 0, 1, 2, 3]),
                                name=None if rng.random() < 0.3 else f"c{i}")
             for i in range(rng.randint(0, 60))]
    one = homogeneous if homogeneous is not None else rng.random() < 0.7
    shapes = [(256, 256)] if one else rng.sample([(256, 256), (128, 128), (256, 64)], rng.randint(2, 3))
    types = [LayoutHardCoreType(a, b, rng.choice([1, 2, 4, 8, 16, 40, 120])) for a, b in shapes]
    if one and rng.random() < 0.3:
        types.append(LayoutHardCoreType(256, 256, rng.choice([0, 3])))  # one shape declared twice
    groups = None
    if rng.random() < 0.3 and specs:
        groups, gid = [], 0
        for _ in specs:
            groups.append(gid if rng.random() < 0.4 else None)
            gid += rng.random() < 0.5
    return specs, types, groups, rng


def _kwargs(rng, groups):
    return dict(allow_coalescing=rng.random() < 0.6, allow_splitting=rng.random() < 0.5, coalescing_group_ids=groups)


class TestTheSplitTakesTheReferenceDecisions:
    @pytest.mark.parametrize("homogeneous", [True, False], ids=["one-shape", "mixed"])
    def test_sub_segments_equal(self, homogeneous):
        for seed in range(250 if homogeneous else 120):
            specs, types, groups, rng = _segment(seed, homogeneous=homogeneous)
            kw = _kwargs(rng, groups)
            got = split_softcores_by_capacity(specs, types, **kw)
            want = ref.reference_split_softcores_by_capacity(specs, types, **kw)
            assert [[id(s) for s in sub] for sub in got] == [[id(s) for s in sub] for sub in want], seed

    def test_the_validated_estimate_equals_the_reference(self):
        for seed in range(150):
            specs, types, groups, rng = _segment(seed)
            kw = dict(allow_coalescing=rng.random() < 0.6, allow_splitting=rng.random() < 0.5, core_types=types,
                      coalescing_group_ids=groups)
            got = estimate_passes_for_layout_validated(specs, sum(t.count for t in types), **kw)
            want = ref.reference_estimate_passes_for_layout_validated(specs, sum(t.count for t in types), **kw)
            assert (got[0], [[id(s) for s in p] for p in got[1]], got[2]) == \
                (want[0], [[id(s) for s in p] for p in want[1]], want[2]), seed


class TestTheCapacityCertificate:
    def test_it_is_sound_on_every_subset(self):
        decided = 0
        for seed in range(500):
            specs, types, _, rng = _segment(seed, homogeneous=True)
            for allow_coalescing in (False, True):
                for allow_splitting in (False, True):
                    bound = capacity_bound(specs, types, allow_coalescing=allow_coalescing,
                                           allow_splitting=allow_splitting)
                    if bound is None or bound.total > bound.count:
                        continue
                    decided += 1
                    for _ in range(3):
                        subset = rng.sample(specs, rng.randint(0, len(specs)))
                        assert ref.reference_pack_layout(
                            softcores=subset, core_types=types, allow_neuron_splitting=allow_splitting,
                            allow_coalescing=allow_coalescing).feasible, seed
        assert decided > 150  # [W04] a wide softcore without declared coalescing is unplaceable: it decides nothing

    def test_mixed_core_shapes_hold_no_certificate(self):
        types = [LayoutHardCoreType(256, 256, 10), LayoutHardCoreType(128, 128, 10)]
        assert capacity_bound([LayoutSoftCoreSpec(8, 8)], types, allow_coalescing=True,
                              allow_splitting=True) is None

    def test_an_unplaceable_softcore_never_certifies(self):
        bound = capacity_bound([LayoutSoftCoreSpec(8, 300)], [LayoutHardCoreType(256, 256, 1000)],
                               allow_coalescing=True, allow_splitting=False)
        assert bound is not None and bound.total > bound.count
        assert not ref.reference_pack_layout(softcores=[LayoutSoftCoreSpec(8, 300)],
                                             core_types=[LayoutHardCoreType(256, 256, 1000)]).feasible


class _CountingPack:
    def __init__(self, monkeypatch):
        self.calls = 0
        original = lp.pack_layout

        def counting(**kw):
            self.calls += 1
            return original(**kw)

        monkeypatch.setattr(lp, "pack_layout", counting)


class TestPacksOnlyWhatTheCertificateLeavesOpen:
    def test_a_certified_segment_packs_nothing(self, monkeypatch):
        counter = _CountingPack(monkeypatch)
        specs = [LayoutSoftCoreSpec(144, 16, latency_tag=i % 3) for i in range(200)]
        out = split_softcores_by_capacity(specs, [LayoutHardCoreType(256, 256, 200)], allow_coalescing=True)
        assert counter.calls == 0
        assert [[id(s) for s in sub] for sub in out] == [[id(s) for s in sub] for sub in
                                                         ref.reference_split_softcores_by_capacity(
                                                             specs, [LayoutHardCoreType(256, 256, 200)],
                                                             allow_coalescing=True)]

    def test_the_estimate_does_not_pack_a_split_pass_again(self, monkeypatch):
        specs = [LayoutSoftCoreSpec(16, 16, latency_tag=i // 100) for i in range(400)]
        types = [LayoutHardCoreType(256, 256, 10)]
        counter = _CountingPack(monkeypatch)
        split = split_softcores_by_capacity(specs, types)
        split_calls = counter.calls
        counter = _CountingPack(monkeypatch)   # a new packer: the class demands measure afresh, as in the split
        n, passes, ok = estimate_passes_for_layout_validated(specs, 10, core_types=types)
        assert len(split) == n >= 2 and ok
        assert counter.calls == split_calls


class TestTheCertificateHoldsOnlyWhereThePackerAgrees:
    """Mutation-found edges of the capacity certificate."""

    def test_one_object_listed_twice_is_packed_as_the_reference_packs_it(self):
        """pack_layout copies a batch with one memo (an unnamed spec is then renamed, which splits the alias), so under
        splitting the pick index meets one named object twice and raises; the split must raise the same, never
        certify such a batch."""
        twice = LayoutSoftCoreSpec(8, 300, latency_tag=0, name="twice")
        specs = [twice, LayoutSoftCoreSpec(8, 8, latency_tag=0), twice]
        types = [LayoutHardCoreType(256, 256, 100)]

        def run(split):
            try:
                return ("ok", [[id(s) for s in sub] for sub in split(specs, types, allow_splitting=True)])
            except Exception as exc:  # noqa: BLE001 - both must fail alike
                return ("raise", type(exc).__name__, str(exc))

        assert run(split_softcores_by_capacity) == run(ref.reference_split_softcores_by_capacity)
        assert run(ref.reference_split_softcores_by_capacity)[0] == "raise"

    def test_a_wider_declared_type_without_cores_takes_no_part_in_the_fusion(self):
        """[W04] The packer keeps a wide softcore whole and fuses cores of a type that has them, as the hard-core
        packer does (no pre-split at the widest declared type): 300 axons fuse three 100-axon cores, so a pool of 3
        packs it, and the certificate counts ceil(300 / 100) = 3."""
        types = [LayoutHardCoreType(100, 64, 3), LayoutHardCoreType(200, 64, 0)]
        specs = [LayoutSoftCoreSpec(300, 10, latency_tag=0)]
        bound = capacity_bound(specs, types, allow_coalescing=True, allow_splitting=False)
        assert bound is not None and bound.total == 3 == bound.count
        kw = dict(core_types=types, allow_coalescing=True)
        assert estimate_passes_for_layout_validated(specs, 3, **kw)[2] is True
        assert ref.reference_estimate_passes_for_layout_validated(specs, 3, **kw)[2] is True

"""U1 planner scaling: the accelerated packing engine takes exactly the reference engine's decisions, at a cost
that no longer grows with the pool or with the cores already used.

The oracle is ``reference_engine`` (the engine before the change, verbatim). Every equivalence test compares whole
traces: each placement (core index, softcore, dims), the used and unused core lists in order, the softcores left,
and the error raised; ``pack_layout`` is compared result for result. The cost tests count operations (predicate
calls, pool iterations, instances built, equality comparisons), never walls, so they are deterministic.
"""

from __future__ import annotations

import random

import pytest

import mimarsinan.mapping.layout.layout_packer as lp
from mimarsinan.mapping.layout.layout_types import (
    LayoutHardCoreInstance,
    LayoutHardCoreType,
    LayoutSoftCoreSpec,
)
from mimarsinan.mapping.packing.canonical import canonical_is_mapping_possible, canonical_split_softcore
from mimarsinan.mapping.packing.core_packing import greedy_pack_softcores, pick_best_softcore
from mimarsinan.mapping.packing.placement_cost import note_resident_bank

from unit.mapping.packing import reference_engine as ref


class Soft:
    """A runtime-like softcore: value equality on its label, optional class, latency and weight bank."""

    eq_calls = 0

    def __init__(self, axons, neurons, label, tg=None, latency=None, bank=None):
        self._a, self._n, self.label = int(axons), int(neurons), label
        if tg is not None:
            self.residency_class_id = tg
        if latency is not None:
            self.latency = latency
        if bank is not None:
            self.weight_bank_id = bank

    def get_input_count(self):
        return self._a

    def get_output_count(self):
        return self._n

    def __eq__(self, other):
        Soft.eq_calls += 1
        return isinstance(other, Soft) and (self._a, self._n, self.label) == (other._a, other._n, other.label)

    def __hash__(self):
        return hash((self._a, self._n, self.label))


class Hard:
    """A runtime-like hard core whose placement sets its class and latency as the real cores do."""

    def __init__(self, axons, neurons, label, tg=None):
        self.axons_per_core = self.available_axons = int(axons)
        self.neurons_per_core = self.available_neurons = int(neurons)
        self.label = label
        self.residency_class_id = tg

    def get_input_count(self):
        return self.axons_per_core

    def get_output_count(self):
        return self.neurons_per_core


class CountingList(list):
    """A list that counts the elements its iterators yield (the identity scans walk it)."""

    def __init__(self, items):
        super().__init__(items)
        self.yielded = 0

    def __iter__(self):
        for item in super().__iter__():
            self.yielded += 1
            yield item


def _custom_feasible(soft, hard):
    s_tg, h_tg = getattr(soft, "residency_class_id", None), getattr(hard, "residency_class_id", None)
    if h_tg is not None and s_tg is not None and s_tg != h_tg:
        return False
    return soft.get_input_count() <= hard.available_axons and soft.get_output_count() <= hard.available_neurons


def _place(trace):
    def place(idx, hard, soft):
        hard.available_axons -= soft.get_input_count()
        hard.available_neurons -= soft.get_output_count()
        if hard.residency_class_id is None:
            hard.residency_class_id = getattr(soft, "residency_class_id", None)
        if getattr(hard, "latency", None) is None and getattr(soft, "latency", None) is not None:
            hard.latency = soft.latency
        note_resident_bank(hard, soft)
        trace.append((idx, hard.label, soft.label, soft.get_input_count(), soft.get_output_count()))

    return place


def _split(soft, available):
    def frag(n, suffix):
        return Soft(soft.get_input_count(), n, soft.label + suffix, tg=getattr(soft, "residency_class_id", None),
                    latency=getattr(soft, "latency", None), bank=getattr(soft, "weight_bank_id", None))

    return canonical_split_softcore(soft, available, make_fragments=lambda *, softcore, first_neurons,
                                    remaining_neurons: (frag(first_neurons, ".a"), frag(remaining_neurons, ".b")))


def _fuse(hards):
    return Hard(sum(h.get_input_count() for h in hards), hards[0].get_output_count(),
                "+".join(h.label for h in hards), tg=getattr(hards[0], "residency_class_id", None))


def _scenario(seed):
    rng = random.Random(seed)
    palette = [(8, 8), (16, 4), (4, 16), (24, 12), (12, 30), (40, 6), (64, 64), (100, 20), (20, 100)]
    n = rng.randint(1, 48)
    softs = []
    for i in range(n):
        a, b = rng.choice(palette) if rng.random() < 0.7 else (rng.randint(0, 90), rng.randint(0, 90))
        softs.append(Soft(a, b, f"s{i}", tg=rng.choice([None, 0, 1, 2]) if rng.random() < 0.8 else None,
                          latency=rng.choice([None, 0, 1]), bank=rng.choice([None, None, 7, 9])))
    types = rng.sample([(32, 32), (64, 64), (128, 32), (32, 128), (48, 48)], rng.randint(1, 3))
    hards = []
    for t, (a, b) in enumerate(types):
        for j in range(rng.randint(0, 24)):
            hards.append(Hard(a, b, f"h{t}.{j}", tg=rng.choice([None, None, 0]) if rng.random() < 0.2 else None))
    rng.shuffle(hards)
    return softs, hards, rng


def _run(engine, seed, *, split, fuse, pick=None, canonical=False, counting=False):
    softs, hards, rng = _scenario(seed)
    if counting:
        hards = CountingList(hards)
    trace, used = [], []
    kwargs = dict(softcores=softs, used_hardcores=used, unused_hardcores=hards,
                  is_mapping_possible=canonical_is_mapping_possible if canonical else _custom_feasible,
                  place=_place(trace), fuse_hardcores=_fuse if fuse else None,
                  split_softcore=_split if split else None, split_threshold=rng.choice([0.0, 0.2, 0.5]))
    if pick is not None:
        kwargs["pick_softcore"] = pick
    if canonical and engine is greedy_pack_softcores:
        kwargs["canonical_feasibility"] = True
    try:
        engine(**kwargs)
        outcome = None
    except RuntimeError as exc:
        outcome = str(exc)
    return {"trace": trace, "outcome": outcome, "used": [(h.label, h.available_axons, h.available_neurons)
                                                          for h in used],
            "unused": [h.label for h in hards], "left": sorted(s.label for s in softs)}


MODES = [dict(split=s, fuse=f) for s in (False, True) for f in (False, True)]


class TestTheEngineTakesTheReferenceDecisions:
    @pytest.mark.parametrize("mode", MODES, ids=lambda m: f"split{int(m['split'])}fuse{int(m['fuse'])}")
    @pytest.mark.parametrize("canonical", [False, True], ids=["custom", "canonical"])
    def test_traces_lists_and_errors_equal(self, mode, canonical):
        for seed in range(160):
            assert _run(greedy_pack_softcores, seed, canonical=canonical, **mode) == \
                _run(ref.reference_greedy_pack_softcores, seed, canonical=canonical, **mode), seed

    @pytest.mark.parametrize("fuse", [False, True])
    def test_the_reference_pick_path_is_unchanged(self, fuse):
        for seed in range(80):
            assert _run(greedy_pack_softcores, seed, split=True, fuse=fuse, pick=pick_best_softcore) == \
                _run(ref.reference_greedy_pack_softcores, seed, split=True, fuse=fuse, pick=pick_best_softcore)


def _specs(seed):
    rng = random.Random(seed)
    palette = [(144, 16), (96, 16), (64, 16), (16, 16), (16, 64), (300, 40), (40, 300), (257, 256), (0, 5), (5, 0)]
    out = []
    for i in range(rng.randint(0, 70)):
        a, b = rng.choice(palette) if rng.random() < 0.75 else (rng.randint(0, 520), rng.randint(0, 300))
        out.append(LayoutSoftCoreSpec(a, b, residency_class_id=rng.choice([0, 0, 1, 2, 3]),
                                      latency_tag=rng.choice([None, 0, 1, 2]),
                                      name=None if rng.random() < 0.5 else f"p{i}"))
    if out and rng.random() < 0.3:
        out.append(out[rng.randrange(len(out))])  # one object twice: deepcopy aliases it once
    types = [LayoutHardCoreType(a, b, rng.choice([0, 1, 3, 12, 40, 200]))
             for a, b in rng.sample([(256, 256), (128, 128), (256, 64), (64, 256), (512, 128)], rng.randint(1, 3))]
    return out, types, rng


def _outcome(pack, **kw):
    try:
        return ("result", pack(**kw))
    except Exception as exc:  # noqa: BLE001 - both engines must fail alike
        return ("raise", type(exc).__name__, str(exc))


class TestPackLayoutEqualsTheReference:
    @pytest.mark.parametrize("split", [False, True])
    @pytest.mark.parametrize("coalesce", [False, True])
    @pytest.mark.parametrize("placements", [False, True])
    def test_results_equal(self, split, coalesce, placements):
        for seed in range(120):
            specs, types, _ = _specs(seed)
            kw = dict(softcores=specs, core_types=types, allow_neuron_splitting=split, allow_coalescing=coalesce,
                      collect_placements=placements)
            assert _outcome(lp.pack_layout, **kw) == _outcome(ref.reference_pack_layout, **kw), seed


class TestTheCostNoLongerScalesWithThePool:
    def test_an_unbounded_pool_builds_only_the_cores_it_uses(self, monkeypatch):
        built = {"n": 0}
        original = LayoutHardCoreInstance.__post_init__

        def counting(self):
            built["n"] += 1
            original(self)

        monkeypatch.setattr(LayoutHardCoreInstance, "__post_init__", counting)
        specs = [LayoutSoftCoreSpec(144, 16, latency_tag=0) for _ in range(40)]
        result = lp.pack_layout(softcores=specs, core_types=[LayoutHardCoreType(256, 256, 200_000)])
        assert result.feasible and result.cores_used == 40
        assert built["n"] <= 4 * 40

    def test_taking_a_core_does_not_walk_the_unused_list(self):
        hards = CountingList([Hard(64, 64, f"h{i}") for i in range(5000)])
        softs = [Soft(48, 48, f"s{i}") for i in range(200)]
        greedy_pack_softcores(softcores=softs, used_hardcores=[], unused_hardcores=hards,
                              is_mapping_possible=_custom_feasible, place=_place([]))
        assert len(hards) == 4800
        assert hards.yielded <= 3 * 5000

    def test_best_fit_reads_one_core_per_state(self, monkeypatch):
        calls = {"n": 0}

        def counting(soft, hard):
            calls["n"] += 1
            return canonical_is_mapping_possible(soft, hard)

        monkeypatch.setattr(lp, "canonical_is_mapping_possible", counting)
        n = 6000
        specs = [LayoutSoftCoreSpec(16, 16, residency_class_id=i % 3, latency_tag=0) for i in range(n)]
        result = lp.pack_layout(softcores=specs, core_types=[LayoutHardCoreType(256, 256, n)])
        assert result.feasible
        assert calls["n"] <= 12 * n

    def test_removing_a_picked_softcore_does_not_scan_the_list(self):
        Soft.eq_calls = 0
        n = 2000
        softs = [Soft(8 + i % 40, 8 + (i * 7) % 40, f"s{i}") for i in range(n)]
        softs.sort(key=lambda s: max(s.get_input_count(), s.get_output_count()))
        greedy_pack_softcores(softcores=softs, used_hardcores=[],
                              unused_hardcores=[Hard(64, 64, f"h{i}") for i in range(n)],
                              is_mapping_possible=_custom_feasible, place=_place([]), split_softcore=_split)
        assert softs == []
        assert Soft.eq_calls <= 20 * n


class TestPlacementsNameTheCoreThatHoldsTheSoftcore:
    """``collect_placements``: each placement resolves to the core that took it, so the placements per core equal that
    core's softcore count, call after call. Fused components stay referenced by the materializer; before, a dead
    component's id could be taken by a later core, which then resolved to an unrelated fused core."""

    @pytest.mark.parametrize("split", [False, True])
    def test_placements_per_core_equal_its_softcore_count(self, split):
        for seed in range(150):
            specs, types, _ = _specs(seed)
            specs = list({id(s): s for s in specs}.values())    # one object listed twice is compared elsewhere
            results = [lp.pack_layout(softcores=specs, core_types=types, allow_neuron_splitting=split,
                                      allow_coalescing=coalesce, collect_placements=True)
                       for coalesce in (False, True) for _ in range(2)]
            for result in results:
                if not result.feasible:
                    continue
                per_core = [0] * result.cores_used
                for _, core in result.placements or ():
                    per_core[core] += 1
                assert per_core == [s.softcore_count for s in result.used_core_snapshots], seed
            assert results[0] == results[1] and results[2] == results[3], seed


class TestTheEdgesTheIndexMustKeep:
    """Mutation-found edges: each case holds a decision the reference takes that a near-miss engine would not."""

    @staticmethod
    def _both(make, **kw):
        """(trace, error, used labels, creation indices of the softcores left) on both engines, canonical predicate."""
        out = []
        for engine, extra in ((greedy_pack_softcores, {"canonical_feasibility": True}),
                              (ref.reference_greedy_pack_softcores, {})):
            softs, used, unused = make()
            made = {id(sc): i for i, sc in enumerate(softs)}
            trace: list = []
            try:
                engine(softcores=softs, used_hardcores=used, unused_hardcores=unused,
                       is_mapping_possible=canonical_is_mapping_possible, place=_place(trace), **kw, **extra)
                raised = None
            except RuntimeError as exc:
                raised = str(exc)
            out.append((trace, raised, [h.label for h in used], [made[id(sc)] for sc in softs]))
        return out

    def test_a_bank_affine_softcore_goes_to_the_core_holding_its_bank(self):
        """Two used cores in one state, the later holding the softcore's bank: affinity picks the later one."""
        def make():
            used = [Hard(32, 32, "h0", tg=0), Hard(32, 32, "h1", tg=0)]
            note_resident_bank(used[1], Soft(1, 1, "seed", bank=7))
            return [Soft(8, 8, "s", tg=0, bank=7)], used, [Hard(32, 32, "spare")]
        new, old = self._both(make)
        assert new == old and old[0][0][1] == "h1"

    def test_a_core_entering_a_class_late_queues_behind_its_class(self):
        """A class-less core that takes a classed softcore joins that class's bucket at its end, so between two cores
        of one state the earlier entrant of the class wins the tie (FIFO picks fix the order)."""
        def make():
            softs = [Soft(15, 15, "s1"), Soft(6, 6, "s2", tg=1), Soft(3, 3, "s3", tg=1), Soft(12, 12, "s4", tg=1),
                     Soft(1, 1, "s5", tg=1)]
            return softs, [], [Hard(20, 20, f"h{i}") for i in range(4)]
        new, old = self._both(make, pick_softcore=lambda remaining: remaining[0])
        assert new == old
        assert [step[1] for step in old[0]] == ["h3", "h2", "h3", "h2", "h2"]

    def test_a_run_out_of_cores_leaves_the_same_softcores_behind(self):
        """Equal-valued distinct softcores: removal takes the first equal one, so a run that runs out of cores leaves
        exactly the reference's objects in the list."""
        def make():
            return [Soft(8, 8, "x") for _ in range(6)], [], [Hard(10, 10, f"h{i}") for i in range(2)]
        new, old = self._both(make, split_softcore=_split)
        assert new == old and old[1] is not None and old[3] == [2, 3, 4, 5]

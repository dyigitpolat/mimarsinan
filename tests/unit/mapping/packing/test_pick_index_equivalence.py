"""Golden equivalence: the incremental pick-index reproduces pick_best_softcore exactly."""

from __future__ import annotations

import random

import pytest

from mimarsinan.mapping.packing.core_packing import (
    greedy_pack_softcores,
    pick_best_softcore,
)
from mimarsinan.mapping.packing.pick_index import PickBestIndex


class Soft:
    def __init__(self, axons, neurons, label, tg=None, latency=None):
        self._a = int(axons)
        self._n = int(neurons)
        self.label = label
        if tg is not None:
            self.threshold_group_id = tg
        if latency is not None:
            self.latency = latency

    def get_input_count(self):
        return self._a

    def get_output_count(self):
        return self._n


class Hard:
    def __init__(self, axons, neurons, tg=None, latency=None):
        self.axons_per_core = int(axons)
        self.neurons_per_core = int(neurons)
        self.available_axons = int(axons)
        self.available_neurons = int(neurons)
        if tg is not None:
            self.threshold_group_id = tg
        if latency is not None:
            self.latency = latency

    def get_input_count(self):
        return self.axons_per_core

    def get_output_count(self):
        return self.neurons_per_core


def _is_mapping_possible(soft, hard):
    s_tg = getattr(soft, "threshold_group_id", None)
    h_tg = getattr(hard, "threshold_group_id", None)
    if h_tg is not None and s_tg is not None and s_tg != h_tg:
        return False
    return (
        soft.get_input_count() <= hard.available_axons
        and soft.get_output_count() <= hard.available_neurons
    )


def _make_place(trace):
    def place(idx, hard, soft):
        hard.available_axons -= soft.get_input_count()
        hard.available_neurons -= soft.get_output_count()
        trace.append((idx, soft.label, soft.get_input_count(), soft.get_output_count()))

    return place


def _split(soft, available_neurons):
    n1 = int(available_neurons)
    n2 = soft.get_output_count() - n1
    return (
        Soft(soft.get_input_count(), n1, soft.label + ".a",
             tg=getattr(soft, "threshold_group_id", None),
             latency=getattr(soft, "latency", None)),
        Soft(soft.get_input_count(), n2, soft.label + ".b",
             tg=getattr(soft, "threshold_group_id", None),
             latency=getattr(soft, "latency", None)),
    )


def _fuse(hardcores):
    axons = sum(h.available_axons for h in hardcores)
    neurons = min(h.available_neurons for h in hardcores)
    return Hard(axons, neurons)


def _scenario(seed, n_soft=40, with_tg=False):
    rng = random.Random(seed)
    softs = [
        Soft(rng.randint(1, 96), rng.randint(1, 96), f"s{i}",
             tg=(rng.choice([0, 1]) if with_tg else None))
        for i in range(n_soft)
    ]
    # Duplicated shapes on purpose: tie-breaking must match max()'s
    # first-occurrence semantics.
    for i in range(0, n_soft, 5):
        softs[i]._a = 64
        softs[i]._n = 64
    hards = [Hard(128, 128, tg=(i % 2 if with_tg else None)) for i in range(64)]
    hards += [Hard(64, 256) for _ in range(32)]
    return softs, hards


def _run(pick, seed, *, split, fuse, with_tg=False):
    softs, hards = _scenario(seed, with_tg=with_tg)
    trace = []
    kwargs = dict(
        softcores=softs,
        used_hardcores=[],
        unused_hardcores=hards,
        is_mapping_possible=_is_mapping_possible,
        place=_make_place(trace),
        fuse_hardcores=_fuse if fuse else None,
        split_softcore=_split if split else None,
    )
    if pick is not None:
        kwargs["pick_softcore"] = pick
    try:
        greedy_pack_softcores(**kwargs)
    except RuntimeError as exc:
        # Infeasible pools must fail identically on both paths.
        trace.append(("raise", str(exc)))
    return trace


class TestPickIndexPackingEquivalence:
    """Default (accelerated) path is trace-identical to the explicit
    pick_best_softcore reference whenever splitting is enabled (the only
    regime where the reference scan used to run)."""

    @pytest.mark.parametrize("seed", range(12))
    def test_split_no_fuse(self, seed):
        assert _run(None, seed, split=True, fuse=False) == _run(
            pick_best_softcore, seed, split=True, fuse=False
        )

    @pytest.mark.parametrize("seed", range(12))
    def test_split_and_fuse(self, seed):
        assert _run(None, seed, split=True, fuse=True) == _run(
            pick_best_softcore, seed, split=True, fuse=True
        )

    @pytest.mark.parametrize("seed", range(6))
    def test_split_with_threshold_groups(self, seed):
        assert _run(None, seed, split=True, fuse=False, with_tg=True) == _run(
            pick_best_softcore, seed, split=True, fuse=False, with_tg=True
        )

    def test_no_split_keeps_the_sort_fast_path(self):
        # Untouched regime: no-split default calls take the pre-existing sorted
        # fast path (its tie order intentionally differs from the scan).
        trace = _run(None, 3, split=False, fuse=False)
        assert trace, "packing produced no placements"


class TestPickBestIndexUnit:
    def test_first_occurrence_tie_break_matches_max(self):
        cores = [Soft(10, 3, "x"), Soft(10, 5, "y"), Soft(4, 10, "z"), Soft(4, 10, "w")]
        idx = PickBestIndex(cores)
        assert idx.pick() is pick_best_softcore(cores)

    def test_appended_equal_key_does_not_displace_earlier(self):
        cores = [Soft(10, 3, "x"), Soft(4, 8, "y")]
        idx = PickBestIndex(cores)
        late = Soft(10, 3, "late")
        cores.append(late)
        idx.add(late)
        assert idx.pick() is pick_best_softcore(cores)

    def test_discard_promotes_next_exactly_like_max(self):
        cores = [Soft(10, 3, "x"), Soft(9, 4, "y"), Soft(4, 9, "z")]
        idx = PickBestIndex(cores)
        picked = idx.pick()
        assert picked is pick_best_softcore(cores)
        cores.remove(picked)
        idx.discard(picked)
        assert idx.pick() is pick_best_softcore(cores)

    def test_random_interleaved_ops_track_reference(self):
        rng = random.Random(7)
        cores = []
        idx = PickBestIndex(cores)
        counter = 0
        for _ in range(300):
            if cores and rng.random() < 0.4:
                victim = idx.pick()
                assert victim is pick_best_softcore(cores)
                cores.remove(victim)
                idx.discard(victim)
            else:
                c = Soft(rng.randint(1, 20), rng.randint(1, 20), f"c{counter}")
                counter += 1
                cores.append(c)
                idx.add(c)
        while cores:
            victim = idx.pick()
            assert victim is pick_best_softcore(cores)
            cores.remove(victim)
            idx.discard(victim)

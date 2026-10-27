"""[F2] The orders of a core graph (``mapping/latency/core_graph``): the topological order, FIFO for the latency
passes and keyed for the residency units (the ready core of the smallest key next, so the key order itself wherever
that order is topological), the strongly connected sets, and the condensation in keyed order."""

from __future__ import annotations

import random
from typing import List

import pytest

from mimarsinan.mapping.latency.core_graph import condensed_order, strongly_connected_sets, topological_order


def _random_dag(rng: random.Random, n: int, p: float) -> List[List[int]]:
    """Per core its sources, over a hidden topological order of the indices."""
    hidden = list(range(n))
    rng.shuffle(hidden)
    rank = {v: k for k, v in enumerate(hidden)}
    return [sorted(u for u in range(n) if rank[u] < rank[v] and rng.random() < p) for v in range(n)]


def _random_graph(rng: random.Random, n: int, p: float) -> List[List[int]]:
    return [sorted(u for u in range(n) if u != v and rng.random() < p) for v in range(n)]


def _is_topological(order, sources) -> bool:
    position = {v: k for k, v in enumerate(order)}
    return sorted(order) == list(range(len(sources))) and all(
        position[int(s)] < position[v] for v, srcs in enumerate(sources) for s in srcs)


def _reaches(sources, n):
    """reach[u] = every core u reaches along reads (u -> its readers)."""
    readers = [[] for _ in range(n)]
    for v, srcs in enumerate(sources):
        for s in srcs:
            readers[int(s)].append(v)
    reach = []
    for u in range(n):
        seen, stack = {u}, [u]
        while stack:
            for w in readers[stack.pop()]:
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        reach.append(seen)
    return reach


class TestTopologicalOrder:
    def test_without_a_key_it_is_the_fifo_order_the_latency_passes_read(self):
        # 1 and 2 read 0, 3 reads both: FIFO from the sources in index order
        assert topological_order([[], [0], [0], [1, 2]]) == [0, 1, 2, 3]
        assert topological_order([[1], [], [1]]) == [1, 0, 2]

    def test_with_a_key_the_ready_core_of_the_smallest_key_comes_next(self):
        # 0 and 2 are ready, 1 reads 2; the key prefers the larger index
        assert topological_order([[], [2], []], key=lambda v: -v) == [2, 1, 0]
        assert topological_order([[], [2], []]) == [0, 2, 1]

    @pytest.mark.parametrize("seed", range(20))
    def test_with_a_key_it_is_the_key_order_wherever_that_order_is_topological(self, seed):
        rng = random.Random(seed)
        n = rng.randint(1, 30)
        sources = [sorted(u for u in range(v) if rng.random() < 0.25) for v in range(n)]  # index order topological
        assert topological_order(sources, key=lambda v: v) == list(range(n))
        assert topological_order(sources, key=lambda v: (v // 3, v)) == list(range(n))

    @pytest.mark.parametrize("seed", range(20))
    def test_every_keyed_order_is_a_topological_order(self, seed):
        rng = random.Random(100 + seed)
        n = rng.randint(1, 30)
        sources = _random_dag(rng, n, 0.2)
        keys = [rng.random() for _ in range(n)]
        assert _is_topological(topological_order(sources, key=keys.__getitem__), sources)
        assert _is_topological(topological_order(sources), sources)

    def test_a_cycle_refuses_with_or_without_a_key(self):
        for key in (None, lambda v: v):
            with pytest.raises(ValueError, match="cycle"):
                topological_order([[1], [0]], key=key)


class TestStronglyConnectedSets:
    def test_a_cycle_is_one_set_and_every_other_core_its_own(self):
        # 0 -> 1 -> 2 -> 0 (each reads the previous), 3 reads 2
        assert sorted(strongly_connected_sets([[2], [0], [1], [2]])) == [[0, 1, 2], [3]]

    def test_an_acyclic_graph_has_singleton_sets(self):
        assert sorted(strongly_connected_sets([[], [0], [0], [1, 2]])) == [[0], [1], [2], [3]]

    @pytest.mark.parametrize("seed", range(30))
    def test_two_cores_share_a_set_exactly_when_each_reaches_the_other(self, seed):
        rng = random.Random(200 + seed)
        n = rng.randint(1, 25)
        sources = _random_graph(rng, n, rng.choice([0.05, 0.1, 0.2]))
        sets = strongly_connected_sets(sources)
        of = {v: k for k, members in enumerate(sets) for v in members}
        assert sorted(of) == list(range(n)) and all(m == sorted(m) for m in sets)
        reach = _reaches(sources, n)
        for u in range(n):
            for v in range(n):
                assert (of[u] == of[v]) == (v in reach[u] and u in reach[v])

    def test_a_long_chain_needs_no_recursion(self):
        n = 20000
        sources = [[n - 1]] + [[v - 1] for v in range(1, n)]                 # one cycle through every core
        assert strongly_connected_sets(sources) == [list(range(n))]


class TestCondensedOrder:
    def test_the_sets_in_a_keyed_topological_order_of_the_condensation(self):
        # {1, 2} a cycle, 0 reads 1, 3 alone: the set of key 1 first, then 0 (ready, key 0), then 3
        assert condensed_order([[1], [2], [1], []], key=lambda v: v) == [[1, 2], [0], [3]]

    def test_a_set_takes_the_smallest_key_of_its_members(self):
        # {0, 3} a cycle (keys 5 and 0), 1 alone (key 1): the set comes first by its member of key 0
        keys = [5, 1, 9, 0]
        assert condensed_order([[3], [], [], [0]], key=keys.__getitem__) == [[0, 3], [1], [2]]

    @pytest.mark.parametrize("seed", range(30))
    def test_it_is_a_topological_order_of_the_sets_and_the_key_order_on_a_dag_in_key_order(self, seed):
        rng = random.Random(300 + seed)
        n = rng.randint(1, 25)
        sources = _random_graph(rng, n, 0.1)
        order = condensed_order(sources, key=lambda v: v)
        assert sorted(v for members in order for v in members) == list(range(n))
        position = {v: k for k, members in enumerate(order) for v in members}
        assert all(position[int(s)] <= position[v] for v, srcs in enumerate(sources) for s in srcs)
        forward = [sorted(u for u in range(v) if rng.random() < 0.2) for v in range(n)]
        assert condensed_order(forward, key=lambda v: v) == [[v] for v in range(n)]

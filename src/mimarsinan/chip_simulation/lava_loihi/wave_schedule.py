"""Longest-path wave scheduling over a neural segment's core dependency graph."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def core_dependency_graph(cores: Sequence[Any]) -> dict[int, list[int]]:
    """Upstream core indices per core, read from its ``core``-kind axon source spans."""
    return {
        idx: sorted({
            int(sp.src_core)
            for sp in core.get_axon_source_spans()
            if sp.kind == "core"
        })
        for idx, core in enumerate(cores)
    }


def wave_levels(deps: Mapping[int, Sequence[int]]) -> list[list[int]]:
    """Partition the dependency DAG into longest-path levels.

    Wave ``k`` holds the cores whose longest upstream chain has length ``k``, so
    every dependency lands in a strictly earlier wave and no wave has an
    internal edge; waves are NEVER derived from core latencies, which
    ``_align_shiftable_cores`` can invert relative to dependency order.
    """
    levels: dict[int, int] = {}
    visiting: set[int] = set()

    def level_of(idx: int) -> int:
        if idx in levels:
            return levels[idx]
        if idx in visiting:
            raise RuntimeError(f"Cycle detected in neural segment at core {idx}")
        visiting.add(idx)
        depth = 1 + max((level_of(dep) for dep in deps[idx]), default=-1)
        visiting.remove(idx)
        levels[idx] = depth
        return depth

    for idx in deps:
        level_of(idx)

    waves: list[list[int]] = [[] for _ in range(max(levels.values(), default=-1) + 1)]
    for idx in sorted(levels):
        waves[levels[idx]].append(idx)
    return waves

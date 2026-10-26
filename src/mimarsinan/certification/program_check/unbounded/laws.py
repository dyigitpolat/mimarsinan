"""[E1 F1] A tile's readout law against the unbounded neuron's, as the function it declares.

``solve_core_law`` stores an identity ADC (one at least as wide as the core's widest accumulator) as the ADC of that
core-wide width, so a tile's identity ADC has other ``adc_levels`` than its unbounded node's while neither clamp ever
binds. Every other field is per column (the multiplier, shift, bias, output grid, the narrow ADC's step and levels)
and must be equal. Where only ``adc_levels`` differ, the laws declare one function when both columns run the identity
ADC (``adc_shift`` 0) at levels no reachable accumulator exceeds: the column's |acc| bound, its data weights' magnitudes
against each row's input code range (a producer neuron's output codes, its law's ``hi``; a raw-input or host row, the
reading core's armed boundary grid), the law solver's ``accumulator_bounds`` read off the program."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict

import numpy as np

from mimarsinan.certification.program_check.unbounded.keys import ON_OWNER, KeyedIR


def output_levels(k: KeyedIR) -> Dict[int, float]:
    """Per neuron code, the magnitude of the output codes its law emits (``max(|lo|, |hi|)``)."""
    out: Dict[int, float] = {}
    for nid, codes in k.codes.items():
        law = getattr(k.by_id[nid], "readout_law", None)
        if law is not None:
            top = np.maximum(np.abs(np.asarray(law.lo, dtype=np.float64)), np.abs(np.asarray(law.hi, dtype=np.float64)))
            out.update((c, v) for c, v in zip(codes.tolist(), top.tolist()) if c != -1)
    return out


def column_bounds(k: KeyedIR, node: Any, columns: np.ndarray, levels: Dict[int, float]) -> np.ndarray:
    """Each column's |acc| bound over its kept data rows (always-on, off and compacted-away rows excluded); NaN where
    a row's range is undeclared."""
    rows = k.rows[int(node.id)]
    sources = k.node_sources(node)
    codes = k.source_codes([sources[r] for r in rows.tolist()])
    on = k.space.owner(ON_OWNER)
    grid = getattr(node, "boundary_grid", None)
    fallback = float(grid.levels) if grid is not None and getattr(grid, "armed", False) else float("nan")
    data = (codes != -1) & (codes != on)
    row = np.array([levels.get(int(c), fallback) for c in codes[data].tolist()], dtype=np.float64)
    weights = np.abs(np.asarray(node.get_core_matrix(k.graph), dtype=np.float64)[rows][data][:, columns])
    live = weights != 0
    return np.where(live, weights * row[:, None], 0.0).sum(axis=0)


def same_function(a: Any, b: Any, bounds_a: np.ndarray, bounds_b: np.ndarray) -> bool:
    """Two column-aligned laws declare one function: every field equal, or only ``adc_levels`` differ on columns that
    run the identity ADC in both, each at levels its column's accumulator bound never exceeds."""
    for f in fields(a):
        if f.name != "adc_levels" and not np.array_equal(getattr(a, f.name), getattr(b, f.name)):
            return False
    la, lb = np.asarray(a.adc_levels), np.asarray(b.adc_levels)
    moved = la != lb
    if not moved.any():
        return True
    identity = (np.asarray(a.adc_shift) == 0) & (np.asarray(b.adc_shift) == 0)
    covered = (la >= bounds_a) & (lb >= bounds_b)
    return bool((identity & covered)[moved].all())

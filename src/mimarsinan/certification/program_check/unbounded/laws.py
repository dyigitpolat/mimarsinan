"""[E1 F1] A tile's readout law against the unbounded neuron's, as the function it declares.

``solve_core_law`` stores an identity ADC (one at least as wide as the core's widest accumulator) as the ADC of that
core-wide width, and [U-CR C-1] ``solve_identity_law`` clamps the identity output law at the core-wide |z| bound, so a
tile carries other ``adc_levels``, ``lo`` and ``hi`` than its unbounded node while no clamp binds. Every field but a
clamp (``CLAMPS``) is per column (the multiplier, shift, bias, output grid, the ADC's step) and must be equal. Where a
clamp differs on a column, the laws declare one function there when both laws' clamps contain that column's reachable
bound (``reach``): the ADC's levels ``rnd_t(acc_bound)``, the output clamp ``rnd_s(|M0| (acc_bound + 2^t) + |B|)``
(``law_magnitude_bound`` per column), with ``acc_bound`` the column's |acc| bound, its data weights' magnitudes
against each row's input code range (a producer neuron's output codes, its law's ``hi``; a raw-input or host row, the
reading core's armed boundary grid), the law solver's ``accumulator_bounds`` read off the program. A column with a
fused operand (a lane law's ``operand_row``) has an output term that bound does not cover: its clamps compare exactly."""

from __future__ import annotations

import math
from dataclasses import fields
from typing import Any, Dict, List, Tuple

import numpy as np

from mimarsinan.certification.program_check.unbounded.keys import ON_OWNER, KeyedIR
from mimarsinan.models.nn.activations.integer_law import law_magnitude_bound

#: The clamp fields of a law: the ADC's levels and the output clamp.
CLAMPS = ("adc_levels", "lo", "hi")


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


def _rounded(value: int, shift: int) -> int:
    """``rnd_s`` of a nonnegative integer (round half away from zero at step ``2^s``): monotone and odd, so the bound
    of ``|rnd_s(z)|`` over ``|z| <= value``."""
    return (value + (1 << (shift - 1))) >> shift if shift > 0 else value


def reach(law: Any, bounds: np.ndarray) -> List[Tuple[float, float]]:
    """Per column, (the ADC codes', the output's) reachable magnitude from the column's |acc| bound; ``inf`` where
    the bound is undeclared (a row with no declared range) or, for the output, where a fused operand enters it."""
    operand = np.asarray(getattr(law, "operand_row", np.full(law.columns, -1))) >= 0
    out: List[Tuple[float, float]] = []
    for j, bound in enumerate(np.asarray(bounds, dtype=np.float64).tolist()):
        if not math.isfinite(bound):
            out.append((math.inf, math.inf))
            continue
        acc, t = int(math.ceil(bound)), int(law.adc_shift[j])
        z = law_magnitude_bound(acc, law.m0[j], law.bias[j], t)
        out.append((_rounded(acc, t), math.inf if operand[j] else _rounded(z, int(law.shift[j]))))
    return out


def _contains(law: Any, name: str, j: int, held: Tuple[float, float]) -> bool:
    value = float(getattr(law, name)[j])
    return value >= held[0] if name == "adc_levels" else value <= -held[1] if name == "lo" else value >= held[1]


def same_function(a: Any, b: Any, bounds_a: np.ndarray, bounds_b: np.ndarray) -> bool:
    """Two column-aligned laws declare one function: every field equal, or [U-CR C-1] only clamps (``CLAMPS``)
    differ, and on each column where one differs both laws' clamps contain the column's reachable bound (``reach``)."""
    for f in fields(a):
        if f.name not in CLAMPS and not np.array_equal(getattr(a, f.name), getattr(b, f.name)):
            return False
    moved = {name: np.flatnonzero(np.asarray(getattr(a, name)) != np.asarray(getattr(b, name))) for name in CLAMPS}
    if not any(columns.size for columns in moved.values()):
        return True
    held_a, held_b = reach(a, bounds_a), reach(b, bounds_b)
    return all(_contains(a, name, j, held_a[j]) and _contains(b, name, j, held_b[j])
               for name, columns in moved.items() for j in columns.tolist())

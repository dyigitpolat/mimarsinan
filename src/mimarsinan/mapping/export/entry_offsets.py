"""Entry offsets in the integer law (WI-5a): an offset grid's term folded into each core's bias, exactly; beside
pruning, the integer image of value 0 is 0 and the rows pruning dropped are read as the model reads them (N7)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, Generic, Hashable, Mapping, Optional, Sequence, Tuple, TypeVar

import numpy as np
import torch

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.ir.core_law import CoreLaw
from mimarsinan.mapping.export.law_rules import adc_step_exponent
from mimarsinan.mapping.pruning.pruning_apply import PRUNED_ROW_MAGNITUDE
from mimarsinan.models.nn.activations.integer_law import (
    FP64_EXACT_BOUND,
    IntegerLawRangeError,
    clamp_reach,
    law_magnitude_bound,
    round_half_away,
)


def core_entry_offset(core) -> float:
    """The offset of the boundary grid a host-fed core reads (0.0 for every other core)."""
    grid = getattr(core, "boundary_grid", None)
    return 0.0 if grid is None or not grid.armed else grid.entry_offset()


def host_row_weight_sums(core: NeuralCore, ir_graph) -> np.ndarray:
    """``Σ_i W_ij`` (int64) over the rows that read a host value (not off, not always-on, not a core) and that the
    deployed program keeps: a bank slice's pruned rows still name their host source, but no accumulator reads them
    (N7: counting them shifted z by ``M0 k Σ W`` against the model)."""
    matrix = np.asarray(core.get_core_matrix(ir_graph), dtype=np.float64)
    nodes = {int(n.id): n for n in ir_graph.nodes}
    sources = core.input_sources.flatten()
    rows = np.asarray([
        not (source.is_off() or source.is_always_on() or isinstance(nodes.get(int(source.node_id)), NeuralCore))
        for source in sources
    ], dtype=bool)
    pruned = getattr(core, "pruned_row_mask", None)
    if pruned is not None and len(pruned) == len(sources):
        rows &= ~np.asarray(pruned, dtype=bool)
    sums = matrix[rows, :].sum(axis=0)
    if not np.array_equal(sums, np.round(sums)):
        raise ValueError(f"core {core.name!r}: the offset fold reads integer weight codes")
    return sums.astype(np.int64)


def whole_step_code(offset: float, step: float) -> Optional[int]:
    """The code ``k`` of value 0 on a grid of values ``code * step - offset`` where the offset is ``k`` whole steps
    (``k step == offset`` in fp64, ``k`` the nearest code, ties away: the grids' own test of value 0), else ``None``."""
    step = float(step)
    if not np.isfinite(step) or step <= 0.0:
        return None
    code = int(round_half_away(torch.tensor(float(offset) / step, dtype=torch.float64)))
    return code if code * step == float(offset) else None


def offset_multipliers(law: CoreLaw, theta: float, offset: float) -> np.ndarray:
    """``M_o`` per column (int64), the multiplier of the offset term ``M_o Σ_i W_i``. Where the offset is ``k`` whole
    steps of the column's input grid, ``M_o = k M0``: a row reading value 0 (code ``k``) adds ``W (M0 k - M_o) = 0`` to
    z, so the law reads the grid as a signed grid centred on its zero code and dropping that row changes nothing (N7).
    Otherwise ``M_o = rnd(2^s o / (θ Δout))``."""
    scaled = np.ldexp(float(offset) / (float(theta) * law.output_step), law.shift)
    rounded = round_half_away(torch.as_tensor(scaled)).numpy().astype(np.int64)
    codes = [whole_step_code(offset, step) for step in np.asarray(law.input_step, dtype=np.float64).reshape(-1)]
    whole = np.asarray([code is not None for code in codes], dtype=bool)
    k = np.asarray([code or 0 for code in codes], dtype=np.int64)
    return np.where(whole, k * np.asarray(law.m0, dtype=np.int64), rounded)


class PrunedOffsetReaderError(ToolchainRefusal, ValueError):
    """A pruned program in which a core reading an offset grid lost rows the model still reads at a code of that grid,
    and the core's law cannot read them as zero: its ADC rounds or clamps the accumulator the model holds, or value 0
    is no code of the grid."""


def dropped_row_magnitude(core: NeuralCore, ir_graph) -> Tuple[np.ndarray, bool]:
    """Per column, ``Σ_i |W_ij|`` (int64) over the rows pruning dropped from the core's accumulator (zeros when none),
    and whether the core's |acc| bounds already count them: a bank slice keeps its pruned rows (off, in the bounds,
    ``core_bias_and_data_weights``); compaction drops an owned matrix's and records their weight
    (``PRUNED_ROW_MAGNITUDE``)."""
    columns = int(core.get_output_count())
    recorded = getattr(core, PRUNED_ROW_MAGNITUDE, None)
    if core.core_matrix is not None:
        magnitude = np.zeros(columns) if recorded is None else np.asarray(recorded, dtype=np.float64)
        counted = False
    else:
        mask = getattr(core, "pruned_row_mask", None)
        matrix = np.asarray(core.get_core_matrix(ir_graph), dtype=np.float64)
        rows = np.asarray(mask if mask is not None and len(mask) == matrix.shape[0] else [False] * matrix.shape[0],
                          dtype=bool)
        magnitude, counted = np.abs(matrix[rows, :]).sum(axis=0), True
    if magnitude.size != columns or not np.array_equal(magnitude, np.round(magnitude)):
        raise ValueError(f"core {core.name!r}: the dropped rows' weight is not one integer per column")
    return magnitude.astype(np.int64), counted


def offset_reader_bounds(core: NeuralCore, ir_graph, bounds: Sequence[int], levels: int) -> Tuple[int, ...]:
    """The |acc| bounds of a core reading an offset grid over every accumulator the model reaches: a row pruning
    dropped still reads a code of the grid in the model (value 0 is code ``k`` there), at most ``levels`` in
    magnitude, so ``levels Σ|W|`` of the dropped rows joins the bounds of an owned matrix (a bank slice's are in them
    already). Unchanged where nothing was dropped."""
    magnitude, counted = dropped_row_magnitude(core, ir_graph)
    if counted or not magnitude.any():
        return tuple(int(b) for b in bounds)
    return tuple(int(b) + int(levels) * int(m) for b, m in zip(bounds, magnitude.tolist()))


def refuse_inexact_offset_readers(readers: Mapping[int, Tuple[NeuralCore, Sequence[int], float]],
                                  laws: Mapping[int, CoreLaw], ir_graph) -> None:
    """Refuse by name a core that reads an offset grid (``readers``: core id to the core, its |acc| bounds over the
    model's accumulators and its offset) and lost rows of nonzero weight to pruning, unless its law reads them as
    zero. The model's accumulator still holds their ``k Σ W``; ``M_o = k M0`` cancels it after the ADC only where the
    ADC reads the accumulator unchanged (step 1, levels at least the bounds) and value 0 is code ``k``."""
    for cid, (core, bounds, offset) in sorted(readers.items()):
        magnitude, _counted = dropped_row_magnitude(core, ir_graph)
        dropped = magnitude != 0
        if not dropped.any():
            continue
        law = laws[cid]
        step = float(np.asarray(law.input_step, dtype=np.float64).reshape(-1)[0])
        code = whole_step_code(offset, step)
        adc_shift = np.asarray(law.adc_shift, dtype=np.int64).reshape(-1)
        levels = np.asarray(law.adc_levels, dtype=np.int64).reshape(-1)
        rounds = dropped & (adc_shift != 0)
        clamps = dropped & (levels < np.asarray(bounds, dtype=np.int64))
        if code is not None and not rounds.any() and not clamps.any():
            continue
        head = (f"core {core.name!r} reads an offset grid (offset {offset:g}, step {step:g}) and pruning dropped rows "
                f"of nonzero weight in {int(dropped.sum())} of {dropped.size} columns, which the model still reads")
        if code is None:
            why = "; value 0 is no code of that grid, so the dropped rows' term is not 0"
        else:
            column = int(np.flatnonzero(rounds | clamps)[0])
            why = (f" at code {code} (value 0); the law's ADC (step 2^{int(adc_shift[column])}, {int(levels[column])} "
                   f"levels, column {column}) does not read the accumulator unchanged over its bound "
                   f"{int(bounds[column])}, so their term {code} Σ W does not cancel against M_o = {code} M0")
        raise PrunedOffsetReaderError(head + why + ": declare the ADC at least as wide as the accumulator, or prune "
                                      "without the offset grid")


def _shift_bias(law: CoreLaw, delta: np.ndarray) -> CoreLaw:
    return CoreLaw(**{f.name: (law.bias + delta if f.name == "bias" else getattr(law, f.name))
                      for f in dataclasses.fields(CoreLaw)})


def fold_entry_offset(law: CoreLaw, core: NeuralCore, ir_graph,
                      acc_bounds: Optional[Sequence[int]] = None) -> CoreLaw:
    """The core's law with the offset term in its bias: ``B - M_o Σ_i W_i`` (the law itself when no offset). The
    real accumulator on codes ``k`` of values ``k Δin - o`` is ``Δin acc_k - o Σ_i W_i``; the term is integer, so a
    conv's border positions and the model's reading (``Σ_i W_i`` as the accumulator of ones) agree bit for bit."""
    offset = core_entry_offset(core)
    if offset == 0.0:
        return law
    term = offset_multipliers(law, float(core.threshold), offset) * host_row_weight_sums(core, ir_graph)
    folded = _shift_bias(law, -term)
    if acc_bounds is not None:
        bound = law_magnitude_bound(list(acc_bounds), law.m0, folded.bias, law.adc_shift)
        if bound >= FP64_EXACT_BOUND:
            raise IntegerLawRangeError(f"core {core.name!r}: the entry offset's term takes |z| to {bound} >= 2^53")
    return folded


_Law = TypeVar("_Law", bound=CoreLaw)


@dataclass(frozen=True)
class FoldRefit(Generic[_Law]):
    """One folded core's refit inputs: its law with the folded bias, its accumulator bounds, its law kind (an
    identity law; a law's clamp ``reach_hi`` when its ADC is narrow, ``reach_lo`` its low code where it is not
    ``-reach_hi``), the fp64 bound's extra term (a lane's operand term), per column the channel it carries (the cores
    the model reads as one law per channel) and [U-LNR] a fused operand term's interval (``terms``; empty: none)."""

    core: str
    law: _Law
    acc_bounds: Tuple[int, ...]
    identity: bool
    reach_hi: Optional[int]
    channels: Tuple[Hashable, ...]
    extra: int = 0
    reach_lo: Optional[int] = None
    terms: Tuple[Tuple[int, int], ...] = ()


def _adc_steps(fold: FoldRefit) -> np.ndarray:
    """Per column, the finest ADC step exponent spanning the accumulators the folded bias leaves unsaturated, the
    law's reach to its clamp on both sides (:func:`clamp_reach`; the solved exponent where the refit moves no ADC
    step)."""
    law = fold.law
    if fold.reach_hi is None:
        return np.asarray(law.adc_shift, dtype=np.int64)
    levels = int(np.asarray(law.adc_levels).reshape(-1)[0])
    lo = -int(fold.reach_hi) if fold.reach_lo is None else int(fold.reach_lo)
    terms = fold.terms or ((0, 0),) * len(fold.acc_bounds)
    return np.asarray([
        adc_step_exponent(min(int(bound), clamp_reach(int(m), int(b), int(s), lo, int(fold.reach_hi), term)), levels,
                          core=fold.core)
        for bound, m, b, s, term in zip(fold.acc_bounds, law.m0, np.asarray(law.bias, dtype=np.int64), law.shift,
                                        terms)],
        dtype=np.int64)


def refit_folded_laws(folds: Mapping[int, FoldRefit[_Law]]) -> Dict[int, _Law]:
    """``{core id: law}``: the integers that read a folded law's bias, re-solved from the folded bias (an offset fold
    moves B after the solve) per channel, not per core. A channel's position cores hold different folded biases (a
    padded border's host rows sum to less), so every carrier of a channel takes the largest of their values: a
    narrow-ADC step exponent (the clamp's reach), so the ADC spans every position's unsaturated
    accumulators, and an identity law's no-clamp bound (widened, never narrowed), the largest folded ``|z|`` bound;
    the model then reads one law per channel. The fp64 exactness bound adds each core's ``extra``. A law whose
    refit moves nothing is returned unchanged; equal results share one object."""
    steps: Dict[Hashable, int] = {}
    proposals = {cid: _adc_steps(fold) for cid, fold in folds.items()}
    for cid, fold in folds.items():
        for key, t in zip(fold.channels, (int(v) for v in proposals[cid])):
            steps[key] = max(steps.get(key, t), t)
    shifts = {cid: np.asarray([steps[key] for key in fold.channels], dtype=np.int64) for cid, fold in folds.items()}
    reach: Dict[Hashable, int] = {}
    for cid, fold in folds.items():
        if fold.identity:
            z = law_magnitude_bound(list(fold.acc_bounds), fold.law.m0, fold.law.bias, shifts[cid])
            for key, hi in zip(fold.channels, np.abs(np.asarray(fold.law.hi, dtype=np.int64)).tolist()):
                reach[key] = max(reach.get(key, 0), int(hi), z)
    out: Dict[int, _Law] = {}
    shared: Dict[tuple, _Law] = {}
    for cid, fold in folds.items():
        law, adc_shift = fold.law, shifts[cid]
        bound = law_magnitude_bound(list(fold.acc_bounds), law.m0, law.bias, adc_shift) + int(fold.extra)
        if bound >= FP64_EXACT_BOUND:
            raise IntegerLawRangeError(f"core {fold.core!r}: the offset term takes |z| to {bound} >= 2^53")
        changes = {}
        if not np.array_equal(adc_shift, law.adc_shift):
            changes["adc_shift"] = adc_shift
        if fold.identity:
            hi = np.asarray([reach[key] for key in fold.channels], dtype=np.int64)
            if not (np.array_equal(hi, law.hi) and np.array_equal(-hi, law.lo)):
                changes.update(lo=-hi, hi=hi)
        key = (id(law),) + tuple(changes[name].tobytes() for name in sorted(changes))
        if key not in shared:
            shared[key] = dataclasses.replace(law, **changes) if changes else law
        out[cid] = shared[key]
    return out


def refit_folded_law(law: _Law, acc_bounds: Sequence[int], *, identity: bool, reach_hi: Optional[int], core: str,
                     extra: int = 0, reach_lo: Optional[int] = None, terms: Tuple[Tuple[int, int], ...] = ()) -> _Law:
    """:func:`refit_folded_laws` of one core whose every column is a channel of its own."""
    channels = tuple(range(law.columns))
    fold = FoldRefit(core, law, tuple(int(b) for b in acc_bounds), identity, reach_hi, channels, int(extra),
                     reach_lo, tuple(terms))
    return refit_folded_laws({0: fold})[0]


def unfold_entry_offset(law: CoreLaw, core: NeuralCore, ir_graph) -> CoreLaw:
    """The per-channel law a folded core was solved from: its bias plus ``M_o Σ_i W_i``."""
    offset = core_entry_offset(core)
    if offset == 0.0:
        return law
    term = offset_multipliers(law, float(core.threshold), offset) * host_row_weight_sums(core, ir_graph)
    return _shift_bias(law, term)


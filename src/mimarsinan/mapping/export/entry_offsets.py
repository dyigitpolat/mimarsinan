"""Entry offsets in the integer law (WI-5a): an offset grid's term folded into each core's bias, exactly."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, Generic, Hashable, Mapping, Optional, Sequence, Tuple, TypeVar

import numpy as np
import torch

from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.ir.core_law import CoreLaw
from mimarsinan.mapping.export.law_rules import adc_step_exponent
from mimarsinan.models.nn.activations.integer_law import (
    FP64_EXACT_BOUND,
    IntegerLawRangeError,
    law_magnitude_bound,
    round_half_away,
    saturation_reach,
)


def core_entry_offset(core) -> float:
    """The offset of the boundary grid a host-fed core reads (0.0 for every other core)."""
    grid = getattr(core, "boundary_grid", None)
    return 0.0 if grid is None or not grid.armed else grid.entry_offset()


def host_row_weight_sums(core: NeuralCore, ir_graph) -> np.ndarray:
    """``Σ_i W_ij`` (int64) over the rows that read a host value (not off, not always-on, not a core)."""
    matrix = np.asarray(core.get_core_matrix(ir_graph), dtype=np.float64)
    nodes = {int(n.id): n for n in ir_graph.nodes}
    rows = [
        not (source.is_off() or source.is_always_on() or isinstance(nodes.get(int(source.node_id)), NeuralCore))
        for source in core.input_sources.flatten()
    ]
    sums = matrix[np.asarray(rows, dtype=bool), :].sum(axis=0)
    if not np.array_equal(sums, np.round(sums)):
        raise ValueError(f"core {core.name!r}: the offset fold reads integer weight codes")
    return sums.astype(np.int64)


def offset_multipliers(law: CoreLaw, theta: float, offset: float) -> np.ndarray:
    """``M_o = rnd(2^s o / (θ Δout))`` per column (int64), the multiplier of the offset term."""
    scaled = np.ldexp(float(offset) / (float(theta) * law.output_step), law.shift)
    return round_half_away(torch.as_tensor(scaled)).numpy().astype(np.int64)


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
    identity law; a symmetric law's clamp ``reach_hi`` when its ADC is narrow), the fp64 bound's extra term (a lane's
    operand term) and, per column, the channel it carries (the cores the model reads as one law per channel)."""

    core: str
    law: _Law
    acc_bounds: Tuple[int, ...]
    identity: bool
    reach_hi: Optional[int]
    channels: Tuple[Hashable, ...]
    extra: int = 0


def _adc_steps(fold: FoldRefit) -> np.ndarray:
    """Per column, the finest ADC step exponent spanning the accumulators the folded bias leaves unsaturated (the
    solved exponent where the refit moves no ADC step)."""
    law = fold.law
    if fold.reach_hi is None:
        return np.asarray(law.adc_shift, dtype=np.int64)
    levels = int(np.asarray(law.adc_levels).reshape(-1)[0])
    return np.asarray([
        adc_step_exponent(min(int(bound), saturation_reach(int(m), int(b), int(s), int(fold.reach_hi))), levels,
                          core=fold.core)
        for bound, m, b, s in zip(fold.acc_bounds, law.m0, np.asarray(law.bias, dtype=np.int64), law.shift)],
        dtype=np.int64)


def refit_folded_laws(folds: Mapping[int, FoldRefit[_Law]]) -> Dict[int, _Law]:
    """``{core id: law}``: the integers that read a folded law's bias, re-solved from the folded bias (an offset fold
    moves B after the solve) per channel, not per core. A channel's position cores hold different folded biases (a
    padded border's host rows sum to less), so every carrier of a channel takes the largest of their values: a
    symmetric law's narrow-ADC step exponent (``reach_hi``), so the ADC spans every position's unsaturated
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
                     extra: int = 0) -> _Law:
    """:func:`refit_folded_laws` of one core whose every column is a channel of its own."""
    channels = tuple(range(law.columns))
    fold = FoldRefit(core, law, tuple(int(b) for b in acc_bounds), identity, reach_hi, channels, int(extra))
    return refit_folded_laws({0: fold})[0]


def unfold_entry_offset(law: CoreLaw, core: NeuralCore, ir_graph) -> CoreLaw:
    """The per-channel law a folded core was solved from: its bias plus ``M_o Σ_i W_i``."""
    offset = core_entry_offset(core)
    if offset == 0.0:
        return law
    term = offset_multipliers(law, float(core.threshold), offset) * host_row_weight_sums(core, ir_graph)
    return _shift_bias(law, term)


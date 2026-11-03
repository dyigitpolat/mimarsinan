"""[U-LMP] Every multiplier path of a law keeps its path, or the law is refused by name.

The output-stage lane computes ``z = M0 f_adc(acc) + M_1 q + B`` and rounds ``z`` once by the core's shift ``s``
(arith_toggle.md section 3.2; ``os_lane.v``: one shift per element). A law therefore has one multiplier path per input
it multiplies at run time: the data path ``M0`` on the accumulator and, per column with a fused operand (class E), the
operand path ``M_1`` on the operand's code. The laws of this tree, by their paths:

- ``law_solver.solve_core_law`` (and the N2 payload cores it solves): the data path alone, at its own largest shift
  within ``mult_bits`` (``law_rules.multiplier_and_shift``), so ``M0`` keeps ``mult_bits`` bits;
- ``output_law.solve_identity_law``: the data path alone, ``M0 = 2^(mult_bits - 1)`` at shift 0, exact by definition
  of its output step;
- ``lanes/law/solve.solve_lane_law``: the data path and, on a fused add's columns, an operand path, at ONE shared shift
  (the largest at which every path's multiplier fits ``mult_bits``); without an operand it is the core law's form;
- the entry-offset folds (``entry_offsets.offset_multipliers``, ``solve.fold_input_offset``): a compile-time term
  ``M_o sum W`` inside ``B``, not a run-time multiplier path (``M_o = k M0`` on a whole-step offset, so it reads the
  data path's own multiplier).

A path whose ratio sits far below another's keeps only the bits the shared shift leaves it: every multiplier scales
with ``2^s`` and the binding path is already at the top of ``mult_bits``, so no shared shift within the declaration
keeps it finer. W08's lanes-on program (16-bit weights, ``mult_bits`` 16) left ``M0`` at 1, 0, 0 on its add-lane cores:
the data path dropped without a name while every twin read the same law.

The rule (:func:`refuse_lost_paths`, read by the lane solver for every law with a fused operand):

- a path whose multiplier is 0 is lost: refused by name;
- a path the sharing starves (its own largest shift exceeds the shared one) must resolve its reach at the shared shift,
  ``X <= 2^s``: ``X`` is the largest input magnitude at which its term can still move an unsaturated output (the data
  path: the accumulators up to their bound that the law does not clamp, the law's clamp reach over the operand's codes,
  plus half the converter's step; an operand path: its largest declared code). The multiplier's own rounding is at most
  1/2, so it moves ``z`` by at most ``X / 2 <= 2^(s-1)``: half an output code over everything the path reaches, no
  coarser than the law's own rounding. Below that floor the core is refused by name (core, perceptron or lane owner,
  path, both ratios, the bits the multiplier keeps, its reach).

Per-path shifts would keep both paths, but no chip declaration's lane form here declares them (the declared lane has
one shift), so they are not taken: the declared hardware never changes silently. A law that keeps every path is the
law it was, integer for integer; every twin reads the same law, so exactness at atol 0 is unchanged by construction.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.export.law_rules import multiplier_and_shift
from mimarsinan.models.nn.activations.integer_law import clamp_reach


class LawMultiplierPathError(ToolchainRefusal, ValueError):
    """A law whose shared shift leaves a multiplier path at 0 or below the precision floor: the path's term would move
    an output by more than half an output code (or vanish), and no shared shift within ``mult_bits`` keeps it."""


def _owner(core) -> str:
    perceptron = getattr(core, "perceptron_index", None)
    if perceptron is not None:
        return f"perceptron {int(perceptron)}"
    owner = getattr(core, "lane_owner", None)
    return "no perceptron" if owner is None else f"lane owner {int(owner)}"


def _term(m1: int, codes: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    if codes is None:
        return (0, 0)
    ends = (int(m1) * int(codes[0]), int(m1) * int(codes[1]))
    return (min(ends), max(ends))


def _refuse(core, path: str, *, m: int, ratio: float, other: str, other_ratio: float, shift: int, mult_bits: int,
            why: str) -> None:
    raise LawMultiplierPathError(
        f"core {core.name!r} ({_owner(core)}): its {path} multiplier is {m} ({int(abs(m)).bit_length()} bits) at the "
        f"shift {shift} it shares with the {other} path (mult_bits {mult_bits}; ratios {path} {ratio:.6g}, {other} "
        f"{other_ratio:.6g}, {path}/{other} 2^{np.log2(ratio / other_ratio):.1f}): {why}. The declared lane rounds "
        "once by one shift, so no shared shift within mult_bits keeps both paths; declare a wider mult_bits, a weight "
        "grid whose threshold brings the ratios closer (fewer weight bits), or the add on the host (onchip_arithmetic "
        "classes without E)")


def refuse_lost_paths(core, *, shift: int, mult_bits: int, ratio0: float, m0: int,
                      ratio1: Sequence[Optional[float]], m1: Sequence[int], bias: Sequence[int],
                      acc_bounds: Sequence[int], adc_shift: Sequence[int], lo: int, hi: int,
                      codes: Sequence[Optional[Tuple[int, int]]]) -> None:
    """Refuse by name a law with a fused operand whose shared ``shift`` loses a multiplier path (module docstring).
    A law without an operand (one path, at its own shift) is never read here."""
    operands = [(j, float(r), int(m)) for j, (r, m) in enumerate(zip(ratio1, m1)) if r is not None]
    if not operands:
        return
    j_top, r_top, _m = max(operands, key=lambda item: item[1])
    if int(m0) == 0:
        _refuse(core, "data", m=0, ratio=ratio0, other="operand", other_ratio=r_top, shift=shift, mult_bits=mult_bits,
                why="the data path is lost (its multiplier rounds to 0), so the core's output no longer reads its "
                    "accumulator")
    if multiplier_and_shift(float(ratio0), mult_bits, core=core.name)[1] > int(shift):
        reach = max(min(int(bound), clamp_reach(int(m0), int(b), int(shift), int(lo), int(hi),
                                                _term(int(mj), cj) if rj is not None else (0, 0)))
                    + ((1 << int(t)) >> 1)
                    for bound, b, t, rj, mj, cj in zip(acc_bounds, bias, adc_shift, ratio1, m1, codes))
        if reach > (1 << int(shift)):
            _refuse(core, "data", m=int(m0), ratio=ratio0, other="operand", other_ratio=r_top, shift=shift,
                    mult_bits=mult_bits,
                    why=f"its rounding moves an output by up to {reach / (1 << (int(shift) + 1)):.3g} output codes "
                        f"over its reach (accumulators up to {reach}, the floor 2^{int(shift)}: half a code)")
    for j, r, m in operands:
        if m == 0:
            _refuse(core, f"operand (column {j})", m=0, ratio=r, other="data", other_ratio=ratio0, shift=shift,
                    mult_bits=mult_bits, why="the operand path is lost (its multiplier rounds to 0), so the fused add "
                                             "no longer reads its operand")
        reach = max(abs(int(v)) for v in (codes[j] or (0, 0)))
        if multiplier_and_shift(r, mult_bits, core=core.name)[1] > int(shift) and reach > (1 << int(shift)):
            _refuse(core, f"operand (column {j})", m=m, ratio=r, other="data", other_ratio=ratio0, shift=shift,
                    mult_bits=mult_bits,
                    why=f"its rounding moves an output by up to {reach / (1 << (int(shift) + 1)):.3g} output codes "
                        f"over its codes (up to {reach}, the floor 2^{int(shift)}: half a code)")

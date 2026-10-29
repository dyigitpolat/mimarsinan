"""``LaneLaw``: a core law with its lane clauses (WI-5e): offset grids and one fused operand per column."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import ClassVar, FrozenSet, Sequence

import numpy as np

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.ir.core_law import _FLOAT_FIELDS, CoreLaw

#: The lane clauses' fields: the value of input code 0 is ``-input_offset`` (the accumulator rows' grid) and of
#: output code 0 ``-output_offset``; ``operand_row`` is the core row carrying the column's fused operand (``-1``:
#: none), read on its own grid (``operand_step``, ``operand_offset``) and entering the law as ``operand_m * q``.
LANE_FIELDS = ("input_offset", "output_offset", "operand_row", "operand_m", "operand_step", "operand_offset")
_LANE_FLOATS = frozenset({"input_offset", "output_offset", "operand_step", "operand_offset"})


#: [W08] The crossbar axon rows a core gives to fused-add operand rows (set by the fusion on the IR core, carried by
#: its soft core and every neuron-split fragment, summed on the hard core it is placed on; absent: none). The census
#: prices them as the lane's operand port, not as crossbar cells, MACs or programming bytes.
OPERAND_AXONS = "lane_operand_axons"


def operand_axons(core) -> int:
    """The axon rows ``core`` gives to fused-add operands (0 without a fused add)."""
    return int(getattr(core, OPERAND_AXONS, 0) or 0)


#: [U-L] The mark of a core whose layout record the fusion wrote after the layout walk, for a core it inserted (a pool
#: lane's or a reduction's, ``lanes/ir/layout_records.py``): the record states the fused program, the walk's predate it.
INSERTED_RECORD = "layout_record_inserted"


def inserted_record(core) -> bool:
    """Whether ``core``'s layout record was written after the walk, by the pass that inserted the core."""
    return bool(getattr(core, INSERTED_RECORD, False))


class LaneLawRowError(ToolchainRefusal, ValueError):
    """A row operation that would drop or move a fused operand's row off the core."""


@dataclass(frozen=True, eq=False)
class LaneLaw(CoreLaw):
    """``y = clamp(rnd_s(M0 f_adc(acc) + M_1 q + B), lo, hi)`` per column, the value ``y Δout - o_out``.

    ``acc`` reads the accumulator rows only: an operand row carries zero weights in the crossbar, so the core's
    accumulator is unchanged by it, and its code enters the lane after the ADC (arith_toggle.md §3.2, one operand).
    """

    input_offset: np.ndarray
    output_offset: np.ndarray
    operand_row: np.ndarray
    operand_m: np.ndarray
    operand_step: np.ndarray
    operand_offset: np.ndarray

    float_fields: ClassVar[FrozenSet[str]] = _FLOAT_FIELDS | _LANE_FLOATS

    @classmethod
    def blank(cls, columns: int) -> "LaneLaw":
        """Unused columns: every output is code 0, with no operand."""
        zeros = np.zeros(int(columns))
        values = {f.name: zeros for f in fields(cls)}
        values["operand_row"] = np.full(int(columns), -1)
        return cls(**values)

    @classmethod
    def of(cls, law: CoreLaw, **lane) -> "LaneLaw":
        """``law`` with lane clauses (each absent clause neutral: no offset, no operand)."""
        if isinstance(law, LaneLaw) and not lane:
            return law
        columns = law.columns
        values = {f.name: getattr(law, f.name) for f in fields(CoreLaw)}
        neutral = {"input_offset": np.zeros(columns), "output_offset": np.zeros(columns),
                   "operand_row": np.full(columns, -1), "operand_m": np.zeros(columns),
                   "operand_step": np.zeros(columns), "operand_offset": np.zeros(columns)}
        if isinstance(law, LaneLaw):
            neutral = {name: getattr(law, name) for name in LANE_FIELDS}
        neutral.update({name: np.broadcast_to(np.asarray(value), (columns,)) for name, value in lane.items()})
        return cls(**values, **neutral)

    @property
    def has_operands(self) -> bool:
        return bool((self.operand_row >= 0).any())

    def shift_rows(self, offset: int) -> "LaneLaw":
        """The law of this core placed ``offset`` rows down a hard core."""
        if not self.has_operands or int(offset) == 0:
            return self
        rows = np.where(self.operand_row >= 0, self.operand_row + int(offset), -1)
        return self.with_operand_rows(rows)

    def with_operand_rows(self, rows: "Sequence[int] | np.ndarray") -> "LaneLaw":
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        values["operand_row"] = np.asarray(rows, dtype=np.int64)
        return type(self)(**values)

    def keep_rows(self, keep: Sequence[int]) -> "LaneLaw":
        """The law after a row compaction keeping ``keep`` (in order); a dropped operand row refuses by name."""
        if not self.has_operands:
            return self
        position = {int(row): index for index, row in enumerate(np.asarray(keep).tolist())}
        missing = sorted({int(r) for r in self.operand_row.tolist() if r >= 0} - set(position))
        if missing:
            raise LaneLawRowError(f"a row compaction drops the fused operand rows {missing[:4]} of a lane law")
        return self.with_operand_rows([position[int(r)] if r >= 0 else -1 for r in self.operand_row.tolist()])

    def __reduce__(self):
        rows = [np.ascontiguousarray(getattr(self, f.name)).view(np.int64) for f in fields(self)]
        return _lane_law_from_buffer, (np.stack(rows).astype("<i8").tobytes(), self.columns)


def _lane_law_from_buffer(buffer: bytes, width: int) -> LaneLaw:
    names = [f.name for f in fields(LaneLaw)]
    matrix = np.frombuffer(buffer, dtype="<i8").reshape(len(names), int(width)).astype(np.int64)
    return LaneLaw(**{
        name: matrix[k].view(np.float64) if name in LaneLaw.float_fields else matrix[k]
        for k, name in enumerate(names)
    })

"""[E6, WI-5e] The laws condition's column comparison: a core's stamped law columns against the identity core's, a lane
law's fused operands by the source each reads.

A fused add's operand row carries zero crossbar weights (no fan-in comparison sees it), and the executor decodes each
column's operand at the row its lane law names (``value_law.lane_rows``), so the row index is the placement's (cost-only:
``merge_core_law`` shifts it by the axon offset, a per-axon split moves it to the physical row) and the function is the
source that row reads. Every other law field is compared as stamped (``stamped_law``); the operand rows by their row
sources (``operand_source``)."""

from __future__ import annotations

import dataclasses
from dataclasses import fields
from typing import Any, List, Optional, Tuple

import numpy as np

from mimarsinan.mapping.support.value_layouts import placement_value

#: The code a law row outside the core's used rows reads (no source either program can name).
NO_ROW = np.iinfo(np.int64).min


def laws_equal(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is b
    return all(np.array_equal(getattr(a, f.name), getattr(b, f.name)) for f in fields(a))


def law_columns(core: Any, lo: int, hi: int) -> Any:
    return placement_value(core, "readout_law", lo, hi)  # [U-PCH3] per neuron, by the core's declaration


def _operands(law: Any, codes: np.ndarray) -> Tuple[Any, Optional[np.ndarray]]:
    """The law with its operand rows reduced to presence (0 where a column has one, -1 where not), and per column the
    source code of its operand row (``NO_ROW`` past the used rows); a plain law unchanged, no operands."""
    if law is None or not hasattr(law, "operand_row"):
        return law, None
    rows = np.asarray(law.operand_row, dtype=np.int64)
    present = rows >= 0
    inside = present & (rows < len(codes))
    sources = np.full(len(rows), -1, dtype=np.int64)
    sources[present] = NO_ROW
    sources[inside] = np.asarray(codes, dtype=np.int64)[rows[inside]]
    return dataclasses.replace(law, operand_row=np.where(present, 0, -1)), sources


def compare_laws(mine_core: Any, mine_codes: np.ndarray, lo: int, hi: int, their_core: Any, their_codes: np.ndarray,
                 their_lo: int, their_hi: int) -> Tuple[bool, List[str]]:
    """Whether either core carries a law over these columns, and the rules its columns break (empty: equal)."""
    mine, theirs = law_columns(mine_core, lo, hi), law_columns(their_core, their_lo, their_hi)
    if mine is None and theirs is None:
        return False, []
    (mine, mine_src), (theirs, their_src) = _operands(mine, mine_codes), _operands(theirs, their_codes)
    broken = [] if laws_equal(mine, theirs) else ["stamped_law"]
    if mine_src is not None and their_src is not None and not np.array_equal(mine_src, their_src):
        broken.append("operand_source")
    return True, broken

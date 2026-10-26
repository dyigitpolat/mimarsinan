"""[E6, N2] The operand banks' predicates (condition ``tiles``): the payload cells every placed neuron reads, the binding
table, and the cores the executor refuses.

A bound bank's placement reads, per sample, the payload's cells ``[bank_axon_range] x [bank_neuron_range]`` at its
block (``fill_operand_blocks``), each row times the source the row's axon reads. Theorem 1's fan-in clause for such a
neuron is then: the same bank, the same bank column, and the same multiset of (row source, bank row) over its live rows
as the identity program's neuron (``payload_cells``); with the static weights compared as any core's (the blocks
cleared, ``view``), the neuron computes the identity's function of the payload for every input. The program's table
must be the IR's (``operand_binding``: the executor writes each bank from it), and no operand block may sit on a fused
core (``operand_on_fused_core``: the executor refuses it by name)."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from mimarsinan.certification.program_check.report import ConditionResult, Violation
from mimarsinan.certification.program_check.unit_forms.operands import program_bindings
from mimarsinan.certification.program_check.view import OFF, Placed, ProgramView

#: Per IR node column: (bank, bank column, the sorted (row source code, bank row) of its live rows).
Cell = Tuple[int, int, Tuple[Tuple[int, int], ...]]


def _record_block(record: Mapping[str, Any], bindings: Mapping[int, Any]) -> Optional[Tuple[int, int, int, int, int]]:
    bank = record.get("weight_bank_id")
    if bank is None or int(bank) not in bindings:
        return None
    return (int(bank), int(record["axon_offset"]), int(record["axons"]), int(tuple(record["bank_axon_range"])[0]),
            int(tuple(record["bank_neuron_range"])[0]))


def _cells(view: ProgramView, p: Placed) -> Optional[Dict[int, Cell]]:
    """The payload cells of placement ``p`` per IR node column, None for a static placement."""
    block = _record_block(p.record, view.bindings or {})
    if block is None:
        return None
    bank, axon_offset, axons, bank_row0, bank_col0 = block
    cv = view.units[p.unit].cores[p.core]
    rows = []
    for r in range(axons):
        row = axon_offset + r
        code = int(cv.codes[row]) if row < len(cv.codes) else -(row + 2)
        if code != int(OFF):
            rows.append((code, bank_row0 + r))
    signature = tuple(sorted(rows))
    return {p.col0 + k: (bank, bank_col0 + k, signature) for k in range(p.cols)}


def _identity_cells(iota: ProgramView) -> Dict[int, Dict[int, Cell]]:
    out: Dict[int, Dict[int, Cell]] = {}
    for node, placed in iota.placements.items():
        for p in placed:
            cells = _cells(iota, p)
            if cells is not None:
                out.setdefault(node, {}).update(cells)
    return out


def check_payloads(delta: ProgramView, iota: ProgramView, tiles: ConditionResult, counts: Dict[str, Any]) -> None:
    """``payload_cells``: every neuron the deployment places on a bound bank reads the identity neuron's payload
    cells, and every identity neuron that reads payload cells is placed on them; ``operand_on_fused_core`` per core
    the executor refuses."""
    theirs = _identity_cells(iota)
    compared, seen = 0, set()
    for node, placed in delta.placements.items():
        for p in placed:
            mine = _cells(delta, p)
            want = theirs.get(node, {})
            columns = range(p.col0, p.col0 + p.cols)
            if mine is None:
                stray = [c for c in columns if c in want]
                if stray:
                    tiles.add(Violation("tiles", "payload_cells", f"node {node} column {stray[0]}",
                                        "the identity program reads payload cells here; the deployment places the "
                                        "neuron on static weights"))
                continue
            tiles.checked += 1
            compared += sum(len(cell[2]) for cell in mine.values())
            seen.update((node, c) for c in mine)
            bad = [c for c in columns if want.get(c) != mine[c]]
            if bad:
                got, ref = mine[bad[0]], want.get(bad[0])
                tiles.add(Violation("tiles", "payload_cells", f"node {node} column {bad[0]}",
                                    f"reads bank {got[0]} column {got[1]} over {len(got[2])} row(s); the identity "
                                    "neuron reads " + ("no payload" if ref is None else
                                                       f"bank {ref[0]} column {ref[1]} over {len(ref[2])} row(s)")
                                    + " (or the rows read other sources)"))
    for node, cells in theirs.items():
        missing = sorted(c for c in cells if (node, c) not in seen)
        if missing and node in delta.placements:
            placed_static = all(_cells(delta, p) is None for p in delta.placements[node])
            if not placed_static:
                tiles.add(Violation("tiles", "payload_cells", f"node {node} column {missing[0]}",
                                    "an identity neuron's payload cells no deployed placement reads"))
    for unit in delta.units:
        for h, cv in enumerate(unit.cores):
            if cv.operand_refused:
                tiles.add(Violation("tiles", "operand_on_fused_core", f"unit {unit.index} ({unit.stage.name}) core {h}",
                                    "an operand block on a fused core: the value executor refuses it by name "
                                    "(OperandPlacementError), so no run computes this core"))
    counts["payload_cells"] = compared


def check_bindings(program: Any, ir_graph: Any, tiles: ConditionResult) -> None:
    """``operand_binding``: the program's binding table is the IR's (the identity builder carries the IR's)."""
    mine, theirs = dict(program_bindings(program)), dict(getattr(ir_graph, "dynamic_bank_bindings", None) or {})
    if not mine and not theirs:
        return
    tiles.checked += 1
    differ = sorted(int(b) for b in set(mine) | set(theirs) if mine.get(b) != theirs.get(b))
    if differ:
        tiles.add(Violation("tiles", "operand_binding", "program dynamic_bank_bindings",
                            f"bank(s) {differ[:6]} bound otherwise than in the IR (or not at all): the value executor "
                            "writes each bound bank from the program's table"))

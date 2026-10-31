"""[U-C4] Cells per stored weight: the array cells one weight of a core type occupies, and the array they make.

A core type's ``max_axons`` x ``max_neurons`` count the weights it holds (the mapping's capacity). A core type whose
stored weight takes several array cells, in the unit its chip description's per-cell rows are declared for, declares
``cells_per_weight`` (absent: 1) and ``weight_cell_rows``, how many of them lie on adjacent rows of the weight's column
(a differential pair of rows: 2); the other ``cells_per_weight / weight_cell_rows`` lie on adjacent columns of its row
(a weight bit-sliced over narrower cells). A weight is a block of cells, so its core's array is the weight capacity
times the block: the cells a weight write writes and wears (the write law), the cells, rows and columns area, row
drivers and converters scale with (the ``*_physical`` quantities), and the core the aggregates' validity geometry is
judged at. The bit-sliced conversion model's ``cells_per_weight`` (ISAAC-like) is the same quantity: a core type
stating another count beside it refuses by name (``conversion_conflict``).

A chip description (a target bundle) declares its geometry in cells, as the chip's paper prints its arrays, and its
cores' weight capacity is that geometry over the block (``weight_capacity``): cells that hold no whole weights are
refused by name. [U-CR C-3] The technology a core type's weights are priced under is the one it names
(``write_physics``, ``compute_physics``), else the run's ``platform_physics_profile`` (the target's). Where that
technology's chip description states a block, the core type states its own block or is refused by name at the run door
(``technology_block_errors``), so a declaration that replaces the chip description's cores, naming the technology or
not, never runs with the block dropped silently. The document door and the wizard do not read the chip descriptions
here (as for ``conversion_conflict``): the run door is where the refusal holds.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from mimarsinan.common.refusal import ToolchainRefusal

CELLS_PER_WEIGHT = "cells_per_weight"
WEIGHT_CELL_ROWS = "weight_cell_rows"


class WeightCellsError(ToolchainRefusal, ValueError):
    """A weight-cell declaration that states no block: a count above 1 without its rows, or rows that do not divide it."""


def cells_per_weight(core: Mapping[str, Any]) -> int:
    """The array cells one stored weight of ``core`` occupies (1 where it declares none)."""
    return int(core.get(CELLS_PER_WEIGHT, 1))


def block_reason(count: Any, rows: Any) -> Optional[str]:
    """Why ``count`` cells with ``rows`` of them on adjacent rows make no block, else None (a value outside its
    schema is the registry's to name)."""
    if not isinstance(count, int) or (rows is not None and not isinstance(rows, int)):
        return None
    if rows is None:
        return None if count <= 1 else (
            f"{CELLS_PER_WEIGHT} {count} states no {WEIGHT_CELL_ROWS}: how many of a weight's cells lie on adjacent "
            "rows of its column (the rest on adjacent columns of its row)")
    if rows < 1 or count < 1 or count % rows:
        return f"{WEIGHT_CELL_ROWS} {rows} does not divide {CELLS_PER_WEIGHT} {count}"
    return None


def block_factors(count: int, rows: Optional[int]) -> Tuple[int, int]:
    """(cell rows, cell columns) of one stored weight: ``rows`` x ``count / rows``, (1, 1) at one cell per weight. A
    count without a block is refused by name (both doors and every chip description refuse it first)."""
    if (reason := block_reason(count, rows)) is not None:
        raise WeightCellsError(reason)
    return (1, 1) if rows is None else (int(rows), int(count) // int(rows))


def cell_factors(core: Mapping[str, Any]) -> Tuple[int, int]:
    """``block_factors`` of one core type's declaration."""
    return block_factors(cells_per_weight(core), core.get(WEIGHT_CELL_ROWS))


def array_extent(core: Mapping[str, Any]) -> Tuple[int, int]:
    """(cell rows, cell columns) of one core: its weight capacity times the block of each weight."""
    rows, columns = cell_factors(core)
    return int(core["max_axons"]) * rows, int(core["max_neurons"]) * columns


def array_totals(cores: Sequence[Mapping[str, Any]]) -> Tuple[int, int, int]:
    """(cell rows, cell columns, cells) of the declared chip over every core type and its count."""
    extents = [(array_extent(core), int(core.get("count", 1))) for core in cores]
    return (sum(r * n for (r, _), n in extents), sum(c * n for (_, c), n in extents),
            sum(r * c * n for (r, c), n in extents))


def weight_cell_errors(cores: Any) -> List[str]:
    """Why a core type's weight cells refuse at the doors, else []: a count above 1 needs its rows, which divide it."""
    if not isinstance(cores, (list, tuple)):
        return []
    return [f"cores[{i}].{reason}" for i, core in enumerate(cores) if isinstance(core, Mapping)
            and (reason := block_reason(core.get(CELLS_PER_WEIGHT, 1), core.get(WEIGHT_CELL_ROWS))) is not None]


def weight_capacity(cores: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """A chip description's cores from its cell geometry: each dimension over the block's cells along it, or the
    reason the declaration contradicts the geometry (a count without its rows, cells holding no whole weights)."""
    if errors := weight_cell_errors(list(cores)):
        return [], "; ".join(errors)
    out: List[Dict[str, Any]] = []
    for core in cores:
        core, block = dict(core), cell_factors(core)
        for key, per, side in zip(("max_axons", "max_neurons"), block, ("rows", "columns")):
            if int(core[key]) % per:
                return [], (f"{CELLS_PER_WEIGHT} {cells_per_weight(core)} in blocks of {block[0]} x {block[1]} cells "
                            f"contradicts the geometry: its {core[key]} cell {side} hold no whole number of weights")
            core[key] = int(core[key]) // per
        out.append(core)
    return out, None


def conversion_conflict(cores: Sequence[Mapping[str, Any]], conversion_model: Optional[Mapping[str, Any]]
                        ) -> Optional[str]:
    """Why a core type's declared count contradicts the conversion model's ``cells_per_weight``, or None."""
    declared = dict(conversion_model or {})
    model = declared.get(CELLS_PER_WEIGHT)
    stated = sorted({cells_per_weight(c) for c in cores if isinstance(c, Mapping) and CELLS_PER_WEIGHT in c})
    if model is None or not stated or stated == [int(model)]:
        return None
    return (f"the core types declare {CELLS_PER_WEIGHT} {', '.join(map(str, stated))} and the physics' "
            f"{declared.get('model')!r} conversion model declares {CELLS_PER_WEIGHT} {model:g}: one stored "
            "weight's cells, stated twice, must agree")


#: [U-CR C-3] The core-type fields that name a technology (a physics profile) a core type's weights are priced under.
TECHNOLOGY_FIELDS = ("write_physics", "compute_physics")


def technology_block_errors(cores: Any, blocks: Mapping[str, Sequence[Tuple[str, int, int]]],
                            profile: Optional[str] = None) -> List[str]:
    """[U-CR C-3] Why a core type whose weights are priced under a technology whose chip description stores a weight in
    a block of several cells (``blocks``: ``target_bundles.technology_blocks``) refuses, else []: it states no block,
    so its geometry would be read as one cell per weight while the technology's weights take several. Each of
    ``TECHNOLOGY_FIELDS`` is priced under the technology the core type names there, else under the run's ``profile``
    (its ``platform_physics_profile``; review M-1). A stated block is the core type's own (the chip description's, or
    another, a bit-sliced pair for one), as ``conversion_conflict`` reads a stated count."""
    if not isinstance(cores, (list, tuple)):
        return []
    errors: List[str] = []
    for i, core in enumerate(cores):
        if not isinstance(core, Mapping) or CELLS_PER_WEIGHT in core:
            continue
        for technology, (named, unnamed) in _technologies(core, profile).items():
            for description, cells, rows in blocks.get(technology, ()):
                errors.append(
                    f"cores[{i}] {_reading(technology, named, unnamed)}, whose chip description (target "
                    f"{description!r}) stores a weight in {cells} cells on {rows} adjacent rows ({CELLS_PER_WEIGHT} "
                    f"{cells}, {WEIGHT_CELL_ROWS} {rows}); the core type states no block, so its "
                    f"{core.get('max_axons')} x {core.get('max_neurons')} would be read as one cell per weight: state "
                    f"its {CELLS_PER_WEIGHT} and {WEIGHT_CELL_ROWS} ({cells} and {rows} for the chip description's, "
                    f"max_axons x max_neurons counting weights: a core of R x C cells holds R/{rows} x "
                    f"C/{cells // rows} of them)")
    return errors


def _technologies(core: Mapping[str, Any], profile: Optional[str]) -> Dict[str, Tuple[List[str], List[str]]]:
    """Per technology a core type is priced under: the fields naming it, and the fields the run's profile stands for."""
    out: Dict[str, Tuple[List[str], List[str]]] = {}
    for field in TECHNOLOGY_FIELDS:
        if isinstance(name := core.get(field), str) and name:
            out.setdefault(name, ([], []))[0].append(field)
        elif profile:
            out.setdefault(str(profile), ([], []))[1].append(field)
    return out


def _reading(technology: str, named: Sequence[str], unnamed: Sequence[str]) -> str:
    said = ["names " + ", ".join(f"{field} {technology!r}" for field in named)] if named else []
    if unnamed:
        said.append(f"{'' if named else 'names '}no {' or '.join(unnamed)}, so the run's platform_physics_profile "
                    f"{technology!r} stands for {'it' if len(unnamed) == 1 else 'them'}")
    return " and ".join(said)


def check_technology_blocks(cores: Sequence[Mapping[str, Any]], blocks: Mapping[str, Sequence[Tuple[str, int, int]]],
                            profile: Optional[str] = None) -> None:
    """[U-CR C-3] Raise ``ValueError`` with ``technology_block_errors``' reasons where a core type priced under a
    technology whose chip description states a block states none (the run door names the declarations, beside the
    conversion model's agreement, ``platform_constraints_resolver``)."""
    if errors := technology_block_errors(list(cores), blocks, profile):
        raise ValueError("; ".join(errors))


def check_conversion_agreement(cores: Sequence[Mapping[str, Any]],
                               conversion_model: Optional[Mapping[str, Any]]) -> None:
    """Raise ``ValueError`` with ``conversion_conflict``'s reason where the two statements disagree (the run door
    names the declarations, ``platform_constraints_resolver``)."""
    if (conflict := conversion_conflict(cores, conversion_model)) is not None:
        raise ValueError(conflict)

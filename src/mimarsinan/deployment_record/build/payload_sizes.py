"""Payload sizing SSOT: exact programmed-weight bytes (cells and a per-axon core's sign register) + compressed span
counts, and the candidate's one programming census over them (``program_census`` re-exports it and its type).

The documented ``weight_bits`` edge-case choice: ``None`` (an undeclared
platform weight width) FAILS LOUD. ``SegmentCoreRecord.params_bytes`` is an
exact integer per the schema (docs/deployment_record_schema.md §2.1), so an
undeclared width cannot size a payload honestly — and every real pipeline
resolves ``weight_bits`` into ``platform_constraints_resolved`` before the
mapping steps run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple

from mimarsinan.mapping.crossbar_utilization import CoreOccupancy
from mimarsinan.mapping.platform.platform_constraints import PER_AXON


def require_weight_bits(weight_bits: Any) -> int:
    """The resolved platform's weight width, validated (None/non-positive raise)."""
    if weight_bits is None:
        raise ValueError(
            "payload sizing requires a declared platform weight width; "
            "'weight_bits' is missing from the resolved platform constraints"
        )
    bits = int(weight_bits)
    if bits <= 0:
        raise ValueError(f"weight_bits must be positive, got {bits}")
    return bits


def params_bytes(cells_used: int, weight_bits: Any) -> int:
    """``ceil(cells_used * weight_bits / 8)`` — exact programmed-payload bytes."""
    bits = require_weight_bits(weight_bits)
    cells = int(cells_used)
    if cells < 0:
        raise ValueError(f"cells_used must be >= 0, got {cells}")
    return -((cells * bits) // -8)


def sign_register_rows(sign_granularity: Any, rows_used: int) -> int:
    """The rows a core's per-row sign register signs: every used row of a per-axon core (one static sign per
    physical row, ODIN's ``SPI_SYN_SIGN``), none on a per-synapse core (each synapse carries its sign in its cells)."""
    return int(rows_used) if sign_granularity == PER_AXON else 0


def sign_register_bytes(sign_rows: int) -> int:
    """``ceil(sign_rows / 8)``: one bit per signed row, rounded up per core (plan section 21 item 11: the per-row sign
    register counts in ``params_bytes``)."""
    rows = int(sign_rows)
    if rows < 0:
        raise ValueError(f"sign_rows must be >= 0, got {rows}")
    return -(rows // -8)


def core_payload_bytes(cells_used: int, weight_bits: Any, sign_rows: int = 0) -> int:
    """One core's programmed payload, the rule every payload site reads: its cells at the weight width
    (``params_bytes``) and its sign register (``sign_register_bytes``, 0 on a per-synapse core)."""
    return params_bytes(cells_used, weight_bits) + sign_register_bytes(sign_rows)


def core_sign_rows(core: Any, occupancy: Optional[CoreOccupancy] = None) -> int:
    """The used rows a built hard core's per-row sign register signs (0 unless the core is of a per-axon type)."""
    occupancy = occupancy or CoreOccupancy.from_hard_core(core)
    return sign_register_rows(getattr(core, "weight_sign_granularity", None), occupancy.axons_used)


def pass_program_bytes(program: Any, weight_bits: Any) -> Tuple[int, ...]:
    """Per core of a candidate pass program, its payload by the one rule (cells and sign register)."""
    signs = tuple(getattr(program, "core_sign_rows", ()) or ())
    return tuple(core_payload_bytes(cells, weight_bits, signs[i] if i < len(signs) else 0)
                 for i, cells in enumerate(program.core_cells))


def core_connectivity_entries(hard_core: Any) -> int:
    """Compressed axon-source span count of one packed hard core.

    Reuses the span SSOT — ``HardCore.get_axon_source_spans()`` caches
    ``compress_spike_sources``, the same run-length encoding behind the
    codegen ``compress_sources_to_spans`` export — never a reimplementation.
    """
    return len(hard_core.get_axon_source_spans())


@dataclass(frozen=True)
class CandidateProgramming:
    """The programming census as the quantity surface consumes it."""

    #: Core initializations: the physical tiles of each pass's evaluated cores (a head's dark cores out).
    segment_cores: int
    #: [R1] Σ over EVERY pass of its evaluated cores' occupied used-rows x used-columns cells — the AS-MAPPED
    #: figure (replicas count once per replica): ``macs``/``cells_used`` at candidate completeness.
    committed_cells: int
    reprogrammed_cores: int
    #: None when the platform declares no weight width — absent, never zero.
    reprogrammed_bytes: Optional[int]
    reprogram_passes: int
    #: Per pass, the physical tiles of every core it allocates (dark ones included); summed: ``cores_allocated``.
    pass_cores: Tuple[int, ...]
    reprogrammed_cells: Tuple[int, ...]


def candidate_programming_census(noc: Any, *, weight_bits: Any) -> CandidateProgramming:
    """[E2] The programming multiplicands of a candidate's pass structure.

    A resident pass contributes its cores to core INIT and nothing else. An undeclared
    ``weight_bits`` cannot size a payload, so the BYTES stay absent while the core counts count.
    Every core count is of physical tiles (a fused core counts its members, ``_tiles``); the payload is
    sized per allocated crossbar (``pass_program_bytes``), as the record sizes ``SegmentCoreRecord.params_bytes``.
    [U-F4] Core initializations and committed cells count each pass's EVALUATED cores (``_evaluated``: a wavefront
    head's dark cores are loaded, never evaluated; every core of any other pass). The written cores, bytes and cells
    and the per-pass cores count every core a pass loads, dark ones included.
    """
    programs = () if noc is None else noc.pass_programs
    written = tuple(program for program in programs if not program.resident)
    payloads = None if weight_bits is None else tuple(
        b for program in written for b in pass_program_bytes(program, weight_bits))
    return CandidateProgramming(
        segment_cores=sum(_tiles(program, _evaluated(program)) for program in programs),
        committed_cells=sum(program.core_cells[i] for program in programs for i in _evaluated(program)),
        reprogrammed_cores=sum(_tiles(program) for program in written),
        reprogrammed_bytes=None if payloads is None else sum(payloads),
        reprogram_passes=len(written),
        pass_cores=tuple(_tiles(program) for program in programs),
        reprogrammed_cells=tuple(cells for program in written for cells in program.core_cells),
    )


def _evaluated(program: Any) -> Tuple[int, ...]:
    """The cores a pass evaluates (``PassProgram.evaluated``); a program that states none evaluates every core."""
    evaluated = getattr(program, "evaluated", None)
    return tuple(evaluated()) if evaluated is not None else tuple(range(len(program.core_cells)))


def _tiles(program: Any, cores: Optional[Iterable[int]] = None) -> int:
    """The physical tiles of a pass program's ``cores`` (all of them by default), the census's one tile accessor:
    per core ``PassProgram.core_tiles`` (a fused core counts its members); a program that states none counts each
    core once."""
    per_core = getattr(program, "core_tiles", None)
    indices = range(len(program.core_cells)) if cores is None else cores
    return sum(1 if per_core is None else int(per_core(i)) for i in indices)
